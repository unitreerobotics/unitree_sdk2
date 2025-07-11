#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>

#include <opencv2/opencv.hpp>  // OpenCV for image processing
#include <curl/curl.h> // HTTP request

#include <unitree/robot/go2/video/video_client.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

#define TOPIC_HIGHSTATE "rt/sportmodestate"

using namespace unitree::common;

unitree_go::msg::dds_::SportModeState_ g_state;

void state_callback(const void* message)
{
    g_state = *(unitree_go::msg::dds_::SportModeState_*)message;
}

void send_to_vllm_server(const std::string& image_path,
    const std::array<float, 3>& pos,
    const std::array<float, 3>& rpy)
{
    CURL *curl;
    CURLcode res;
    curl_mime *form = nullptr;
    curl_mimepart *field = nullptr;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (curl) {
        form = curl_mime_init(curl);

        // 이미지 전송
        field = curl_mime_addpart(form);
        curl_mime_name(field, "image");
        curl_mime_filedata(field, image_path.c_str());

        // 위치 (position)
        std::string pos_json = "{\"x\":" + std::to_string(pos[0]) +
          ",\"y\":" + std::to_string(pos[1]) +
          ",\"z\":" + std::to_string(pos[2]) + "}";
        field = curl_mime_addpart(form);
        curl_mime_name(field, "position");
        curl_mime_data(field, pos_json.c_str(), CURL_ZERO_TERMINATED);

        // RPY (roll/pitch/yaw)
        std::string rpy_json = "{\"roll\":" + std::to_string(rpy[0]) +
          ",\"pitch\":" + std::to_string(rpy[1]) +
          ",\"yaw\":" + std::to_string(rpy[2]) + "}";
        field = curl_mime_addpart(form);
        curl_mime_name(field, "rpy");
        curl_mime_data(field, rpy_json.c_str(), CURL_ZERO_TERMINATED);

        // send to server (http://127.0.0.1:5000/analyze)
        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5000/analyze");
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "[curl error] " << curl_easy_strerror(res) << std::endl;
        }

        curl_mime_free(form);
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}
    

int main()
{
    // Init SDK
    unitree::robot::ChannelFactory::Instance()->Init(0);

    // Init video
    unitree::robot::go2::VideoClient video_client;
    video_client.SetTimeout(1.0f);
    video_client.Init();

    // Init state subscriber
    auto sub = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(TOPIC_HIGHSTATE);
    sub->InitChannel(state_callback, 1);

    std::vector<uint8_t> image_sample;
    int ret;

    while (true)
    {
        auto start_time = std::chrono::steady_clock::now();  // frame start time
        ret = video_client.GetImageSample(image_sample);

        if (ret == 0 && !image_sample.empty()) {
            std::string raw_name = "raw_image.jpg";
            std::string processed_name = "last_image.jpg";

            // Save raw image
            std::ofstream image_file(raw_name, std::ios::binary);
            if (image_file.is_open()) {
                image_file.write(reinterpret_cast<const char*>(image_sample.data()), image_sample.size());
                image_file.close();
            } else {
                std::cerr << "Error: Failed to save raw image." << std::endl;
                continue;
            }

            // Load with OpenCV
            cv::Mat img = cv::imread(raw_name);
            if (img.empty()) {
                std::cerr << "Error: Failed to read image with OpenCV." << std::endl;
                continue;
            }

            // Center crop
            int h = img.rows;
            int w = img.cols;
            int crop_size = std::min(h, w);
            int top = (h - crop_size) / 2;
            int left = (w - crop_size) / 2;
            cv::Mat cropped = img(cv::Rect(left, top, crop_size, crop_size));

            // Resize to 320x320
            cv::Mat resized;
            cv::resize(cropped, resized, cv::Size(320, 320));

            // Save final image
            cv::imwrite(processed_name, resized);

            // Pose info
            const auto& pos = g_state.position();
            const auto& rpy = g_state.imu_state().rpy();

            if (pos.size() >= 3 && rpy.size() >= 3) {
                std::cout << "Image saved as " << processed_name << std::endl;
                std::cout << "Position: x=" << pos[0] << ", y=" << pos[1] << ", z=" << pos[2] << std::endl;
                std::cout << "RPY: roll=" << rpy[0] << ", pitch=" << rpy[1] << ", yaw=" << rpy[2] << std::endl;
            } else {
                std::cout << "Waiting for valid pose data..." << std::endl;
            }

            auto end_time = std::chrono::steady_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;

            // send to server
            std::array<float, 3> pos_arr = {pos[0], pos[1], pos[2]};
            std::array<float, 3> rpy_arr = {rpy[0], rpy[1], rpy[2]};
            send_to_vllm_server(processed_name, pos_arr, rpy_arr);
        }

        usleep(50000);  // 0.05 seconds
    }

    return 0;
}
