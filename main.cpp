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
    cv::Mat templateR = imread("../image/R.png", IMREAD_GRAYSCALE); // 加载R形模板
    cv::Mat detectedR; // 存储检测到的R形图像

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

        // 模板匹配以检测“R”形图标
        cv::matchTemplate(graySrc, templateR, detectedR, TM_CCOEFF_NORMED);
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(detectedR, &minVal, &maxVal, &minLoc, &maxLoc);
        if (maxVal > 0.8) // 设定一个阈值
        {
            cv::Point2f centerR(maxLoc.x + templateR.cols / 2, maxLoc.y + templateR.rows / 2);
            std::cout << "Detected R at: (" << centerR.x << ", " << centerR.y << ")" << std::endl;

            // 输出面积最小的长方形中心坐标
            if (minArea < std::numeric_limits<double>::max())
            {
                std::cout << "Min Rectangle Center: (" << minRectInfo.center.x << ", " << minRectInfo.center.y << ")" << std::endl;

                // 输出相对坐标
                cv::Point2f relativePosition = minRectInfo.center - centerR;
                std::cout << "Relative Position: (" << relativePosition.x << ", " << relativePosition.y << ")" << std::endl;
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
