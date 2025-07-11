#include <unitree/robot/go2/video/video_client.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <unistd.h>  // for sleep
#include <opencv2/opencv.hpp>  // OpenCV for image processing

int main()
{
    unitree::robot::ChannelFactory::Instance()->Init(0);
    unitree::robot::go2::VideoClient video_client;
    video_client.SetTimeout(1.0f);
    video_client.Init();

    std::vector<uint8_t> image_sample;
    int ret;

    while (true)
    {
        ret = video_client.GetImageSample(image_sample);

        if (ret == 0) {
            std::string raw_name = "raw_image.jpg";
            std::string processed_name = "last_image.jpg";

            // Save raw image
            std::ofstream image_file(raw_name, std::ios::binary);
            if (image_file.is_open()) {
                image_file.write(reinterpret_cast<const char*>(image_sample.data()), image_sample.size());
                image_file.close();
                std::cout << "Raw image saved as " << raw_name << std::endl;
            } else {
                std::cerr << "Error: Failed to save raw image." << std::endl;
                continue;
            }

            // Load image with OpenCV
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
            cv::Rect roi(left, top, crop_size, crop_size);
            cv::Mat cropped = img(roi);

            // Resize to 320x320
            cv::Mat resized;
            cv::resize(cropped, resized, cv::Size(320, 320));

            // Save final image
            cv::imwrite(processed_name, resized);
            std::cout << "Processed image saved as " << processed_name << std::endl;
        }

        usleep(50000);  // 0.05 seconds = 50,000 microseconds
    }

    return 0;
}
