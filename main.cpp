#include "windmill.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <limits>

using namespace std;
using namespace cv;

struct RectangleInfo
{
    cv::Point2f center;
    int id;
    int survivalCount; // 生存计数器
    double area;       // 面积
};

int main()
{
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    WINDMILL::WindMill wm(t.count());
    cv::Mat src;

    std::map<int, RectangleInfo> rectangles; // 存储已检测到的长方形
    int nextID = 0;                          // 下一个可用ID

    while (true)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        src = wm.getMat((double)t.count() / 1000);
        cv::Mat graySrc, edges;

        // 转为灰度图像
        cv::cvtColor(src, graySrc, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(graySrc, graySrc, cv::Size(5, 5), 0);
        cv::Canny(graySrc, edges, 50, 150);
        imshow("Edges", edges);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<RectangleInfo> detectedRectangles; // 当前帧检测到的长方形
        double minArea = std::numeric_limits<double>::max();
        RectangleInfo minRectInfo;

        for (const auto &contour : contours)
        {
            cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
            double area = rotatedRect.size.width * rotatedRect.size.height;

            if (area > 400)
            {
                RectangleInfo rectInfo;
                rectInfo.center = rotatedRect.center;
                rectInfo.survivalCount = 0; // 初始化生存计数器
                rectInfo.area = area;       // 记录面积

                // 检查是否已经有相似的长方形
                bool found = false;
                for (const auto &entry : rectangles)
                {
                    if (cv::norm(rectInfo.center - entry.second.center) < 30)
                    {
                        rectInfo.id = entry.second.id;
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    rectInfo.id = nextID++;
                }

                detectedRectangles.push_back(rectInfo);

                // 更新最小面积的长方形
                if (area < minArea)
                {
                    minArea = area;
                    minRectInfo = rectInfo;
                }

                cv::Point2f rectPoints[4];
                rotatedRect.points(rectPoints);
                for (int i = 0; i < 4; i++)
                {
                    cv::line(src, rectPoints[i], rectPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                }
                cv::circle(src, rectInfo.center, 5, cv::Scalar(255, 0, 0), -1);
                std::string idText = "ID: " + std::to_string(rectInfo.id);
                cv::putText(src, idText, rectInfo.center + cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

                std::cout << "Rectangle Detected, ID: " << rectInfo.id << ", Center: (" << rectInfo.center.x << ", " << rectInfo.center.y << ")" << std::endl;
            }
        }

        // 更新已检测到的长方形
        rectangles.clear();
        for (const auto &detected : detectedRectangles)
        {
            rectangles[detected.id] = detected;
            rectangles[detected.id].survivalCount = 0; // 重置生存计数器
        }

        // 更新生存计数器和移除超时长方形
        for (auto it = rectangles.begin(); it != rectangles.end();)
        {
            it->second.survivalCount++;
            if (it->second.survivalCount > 10)
            { // 超过10帧移除
                it = rectangles.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // 绘制最小面积的长方形的蓝色外框
        if (minArea < std::numeric_limits<double>::max())
        {
            cv::RotatedRect minRotatedRect = cv::minAreaRect(contours[std::distance(contours.begin(), std::find_if(contours.begin(), contours.end(), [&minRectInfo](const std::vector<cv::Point> &c)
                                                                                                                   { return cv::norm(minRectInfo.center - cv::minAreaRect(c).center) < 30; }))]); // 获取对应的轮廓
            cv::Point2f minRectPoints[4];
            minRotatedRect.points(minRectPoints);
            for (int i = 0; i < 4; i++)
            {
                cv::line(src, minRectPoints[i], minRectPoints[(i + 1) % 4], cv::Scalar(255, 0, 0), 2); // 使用蓝色外框
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
        std::cout << "Recognition Time: " << elapsed_time.count() << " ms" << std::endl;

        imshow("Windmill", src);
        if (waitKey(1) >= 0)
            break;
    }

    return 0;
}
