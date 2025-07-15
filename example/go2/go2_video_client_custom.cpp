#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <curl/curl.h>

#include <unitree/robot/go2/video/video_client.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/LowState_.hpp>

#define TOPIC_HIGHSTATE "rt/sportmodestate"
#define TOPIC_LOWSTATE "rt/lowstate"

using namespace unitree::common;

unitree_go::msg::dds_::SportModeState_ g_state;
unitree_go::msg::dds_::LowState_ g_low_state;

// for position, rpy
void state_callback(const void* message) {
    g_state = *(unitree_go::msg::dds_::SportModeState_*)message;
}

// for joints
void low_state_callback(const void* message) {
    g_low_state = *(unitree_go::msg::dds_::LowState_*)message;
}

std::string generate_motor_json(const std::array<unitree_go::msg::dds_::MotorState_, 12>& motors) {
    std::string json = "[";
    for (int i = 0; i < 12; ++i) {
        json += "{\"q\":" + std::to_string(motors[i].q()) +
                ",\"dq\":" + std::to_string(motors[i].dq()) +
                ",\"tau\":" + std::to_string(motors[i].tau_est()) + "}";
        if (i != 11) json += ",";
    }
    json += "]";
    return json;
}

void send_to_vllm_server(const std::string& image_path,
                         const std::array<float, 3>& pos,
                         const std::array<float, 3>& rpy,
                         const std::array<unitree_go::msg::dds_::MotorState_, 12>& joints)
{
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[curl error] Failed to initialize CURL." << std::endl;
        return;
    }

    curl_mime* form = curl_mime_init(curl);

    auto add_field = [&](const char* name, const char* data) {
        curl_mimepart* field = curl_mime_addpart(form);
        curl_mime_name(field, name);
        curl_mime_data(field, data, CURL_ZERO_TERMINATED);
    };

    auto add_json_field = [&](const char* name, const std::string& json) {
        add_field(name, json.c_str());
    };

    // Image
    curl_mimepart* field = curl_mime_addpart(form);
    curl_mime_name(field, "image");
    curl_mime_filedata(field, image_path.c_str());

    // Position & RPY
    add_json_field("position", "{\"x\":" + std::to_string(pos[0]) + ",\"y\":" + std::to_string(pos[1]) + ",\"z\":" + std::to_string(pos[2]) + "}");
    add_json_field("rpy", "{\"roll\":" + std::to_string(rpy[0]) + ",\"pitch\":" + std::to_string(rpy[1]) + ",\"yaw\":" + std::to_string(rpy[2]) + "}");

    // Joints
    add_json_field("joints", generate_motor_json(joints));

    // CURL options
    curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5000/analyze");
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[curl error] " << curl_easy_strerror(res) << std::endl;
    }

    curl_mime_free(form);
    curl_easy_cleanup(curl);
}

cv::Mat process_image(const std::vector<uint8_t>& image_data, const std::string& save_path) {
    std::string tmp_path = "tmp_raw.jpg";
    std::ofstream file(tmp_path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(image_data.data()), image_data.size());
    file.close();

    cv::Mat img = cv::imread(tmp_path);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image with OpenCV.");
    }

    int size = std::min(img.rows, img.cols);
    cv::Rect roi((img.cols - size) / 2, (img.rows - size) / 2, size, size);

    cv::Mat cropped = img(roi);
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(320, 320));
    cv::imwrite(save_path, resized);

    return resized;
}

void print_pose_info(const std::array<float, 3>& pos, const std::array<float, 3>& rpy) {
    std::cout << "Position: x=" << pos[0] << ", y=" << pos[1] << ", z=" << pos[2] << std::endl;
    std::cout << "RPY: roll=" << rpy[0] << ", pitch=" << rpy[1] << ", yaw=" << rpy[2] << std::endl;
}

void print_motor_info(const std::array<unitree_go::msg::dds_::MotorState_, 12>& motors) {
    const char* leg_names[4] = {"FR", "FL", "RR", "RL"};
    const char* joint_names[3] = {"Hip", "Thigh", "Calf"};

    for (int leg = 0; leg < 4; ++leg) {
        for (int j = 0; j < 3; ++j) {
            int idx = leg * 3 + j;
            std::cout << leg_names[leg] << "_" << joint_names[j] 
                      << " | q: " << motors[idx].q()
                      << " dq: " << motors[idx].dq()
                      << " tau: " << motors[idx].tau_est() << std::endl;
        }
    }
}


int main()
{
    unitree::robot::ChannelFactory::Instance()->Init(0);

    unitree::robot::go2::VideoClient video_client;
    video_client.SetTimeout(1.0f);
    video_client.Init();

    // subscribers
    auto high_sub = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(TOPIC_HIGHSTATE);
    high_sub->InitChannel(state_callback, 1);

    auto low_sub = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(TOPIC_LOWSTATE);
    low_sub->InitChannel(low_state_callback, 1);

    std::vector<uint8_t> image_sample;

    while (true)
    {
        auto start = std::chrono::steady_clock::now();
        if (video_client.GetImageSample(image_sample) != 0 || image_sample.empty()) {
            usleep(50000);
            continue;
        }

        try {
            std::string output_image = "processed_image.jpg";
            process_image(image_sample, output_image);

            const auto& pos_vec = g_state.position();
            const auto& rpy_vec = g_state.imu_state().rpy();
            const auto& motors = g_low_state.motor_state();

            if (pos_vec.size() < 3 || rpy_vec.size() < 3) {
                std::cout << "Waiting for valid pose data..." << std::endl;
                usleep(50000);
                continue;
            }

            std::array<float, 3> pos = {pos_vec[0], pos_vec[1], pos_vec[2]};
            std::array<float, 3> rpy = {rpy_vec[0], rpy_vec[1], rpy_vec[2]};
            std::array<unitree_go::msg::dds_::MotorState_, 12> joints;
            for (int i = 0; i < 12; ++i) joints[i] = motors[i];

            print_pose_info(pos, rpy);
            print_motor_info(joints);

            send_to_vllm_server(output_image, pos, rpy, joints);

            auto end = std::chrono::steady_clock::now();
            std::cout << "Elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[Error] " << e.what() << std::endl;
        }

        usleep(50000);
    }

    return 0;
}
