#include "windmill.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;
using namespace cv;

int main()
{
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    WINDMILL::WindMill wm(t.count());
    cv::Mat src;

    while (true)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        src = wm.getMat((double)t.count() / 1000);
        cv::Mat graySrc, edges, mask;

        // 转为灰度图像
        cv::cvtColor(src, graySrc, cv::COLOR_BGR2GRAY);
        
        // 使用Canny边缘检测
        cv::Canny(graySrc, edges, 50, 150);

        // 寻找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 遍历轮廓，识别叶片
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 100) { // 根据实际情况调整面积阈值
                cv::Rect boundingBox = cv::boundingRect(contour);
                cv::rectangle(src, boundingBox, cv::Scalar(0, 255, 0), 2); // 绘制边界框

                // 记录角度等信息
                double angle = atan2(boundingBox.y + boundingBox.height / 2 - src.rows / 2,
                                      boundingBox.x + boundingBox.width / 2 - src.cols / 2);
                std::cout << "Target Detected, Angle: " << angle << std::endl;
            }
        }

        // Output recognition time
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
        std::cout << "Recognition Time: " << elapsed_time.count() << " ms" << std::endl;

        imshow("Windmill", src);
        if (waitKey(1) >= 0) break;
    }

    return 0;
}
