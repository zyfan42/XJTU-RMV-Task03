#include "windmill.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <limits>
#include <ceres/ceres.h>
#include <fstream>

using namespace std;
using namespace cv;

struct RectangleInfo
{
    cv::Point2f center;
    int id;
    int survivalCount; // 生存计数器
    double area;       // 面积
};

// 旋转余序类，用于ceres编写余序方程
struct SineResidual
{
    SineResidual(double x, double y)
        : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T *const params, T *residual) const
    {
        T A = params[0];
        T omega = params[1];
        T phi = params[2];
        // 模型 y = A * sin(omega * x + phi)
        residual[0] = y_ - (A * ceres::sin(omega * T(x_) + phi));
        return true;
    }

private:
    const double x_;
    const double y_;
};

// 使用简单的滑动平均对观测数据进行滤波和平滑
std::vector<std::pair<double, double>> smoothObservations(const std::vector<std::pair<double, double>> &observations, int windowSize)
{
    std::vector<std::pair<double, double>> smoothedObservations;
    int n = observations.size();
    for (int i = 0; i < n; ++i)
    {
        double sumY = 0.0;
        int count = 0;
        for (int j = i - windowSize / 2; j <= i + windowSize / 2; ++j)
        {
            if (j >= 0 && j < n)
            {
                sumY += observations[j].second;
                count++;
            }
        }
        double smoothedY = sumY / count;
        smoothedObservations.emplace_back(observations[i].first, smoothedY);
    }
    return smoothedObservations;
}

void fitWindmillRotation(const std::vector<std::pair<double, double>> &observations)
{
    // 对观测数据进行平滑处理
    int windowSize = 5;
    std::vector<std::pair<double, double>> smoothedObservations = smoothObservations(observations, windowSize);

    // 初始化参数
    double params[3] = {1.0, 1.0, 0.0}; // 使用自动配置的初始值

    // 创建Ceres问题对象
    ceres::Problem problem;
    for (const auto &obs : smoothedObservations)
    {
        double x = obs.first;
        double y = obs.second;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SineResidual, 1, 3>(
                new SineResidual(x, y)),
            nullptr,
            params);
    }

    // 设置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 20;              // 增加最大迭代次数以确保非线性拟合的收敛
    options.minimizer_type = ceres::TRUST_REGION; // 使用信赖域算法以适应非线性拟合
    options.initial_trust_region_radius = 1.0;    // 设置初始信赖域半径
    options.function_tolerance = 1e-6;            // 减小收敛阈值，提高拟合精度
    options.gradient_tolerance = 1e-10;           // 增加步长，改善初值和真值差异较大的情况

    // 运行求解器并输出结果
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << "Estimated Parameters: A = " << params[0] << ", omega = " << params[1]
              << ", phi = " << params[2] << "\n";
}

int main()
{
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    long long initialTime = t.count();
    WINDMILL::WindMill wm((t.count() - initialTime));
    cv::Mat src;

    std::map<int, RectangleInfo> rectangles;                        // 存储已检测到的长方形
    int nextID = 0;                                                 // 下一个可用ID
    cv::Mat templateR = imread("../image/R.png", IMREAD_GRAYSCALE); // 加载R形模板
    cv::Mat detectedR;                                              // 存储检测到的R形图像

    std::vector<std::pair<double, double>> observations; // 用于旋转观测的时间和角速度

    int frame_count = 0;

    // 实时相应的信息
    cv::Point2f prevRelativePosition;
    bool hasPrevData = false;
    double prevTime = 0.0;
    double prevAngle = 0.0;

    // 打开CSV文件以输出观测值（可注释掉）
    std::ofstream csvFile("observations.csv");
    if (csvFile.is_open())
    {
        csvFile << "Time,AngularVelocity\n";
    }

    while (frame_count < 600)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        src = wm.getMat((double)t.count());
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

            if (area > 10000)
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
        if (maxVal > 0.8) // 设定一个问题
        {
            cv::Point2f centerR(maxLoc.x + templateR.cols / 2, maxLoc.y + templateR.rows / 2);
            std::cout << "Detected R at: (" << centerR.x << ", " << centerR.y << ")" << std::endl;

            // 输出面积最小的长方形中心坐标
            cv::circle(src, centerR, 5, Scalar(255, 0, 0), -1); // 标出R的中心
            if (minArea < std::numeric_limits<double>::max())
            {
                std::cout << "Min Rectangle Center: (" << minRectInfo.center.x << ", " << minRectInfo.center.y << ")" << std::endl;
                cv::circle(src, minRectInfo.center, 5, Scalar(255, 0, 0), -1); // 标出最小矩形的中心
                cv::Point2f vertices[4];
                cv::RotatedRect minRotatedRect = cv::minAreaRect(cv::Mat(contours[0]));
                minRotatedRect.points(vertices);
                for (int i = 0; i < 4; ++i)
                    line(src, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2); // 框出最小矩形

                // 计算当前相对坐标
                cv::Point2f relativePosition = minRectInfo.center - centerR;
                std::cout << "Relative Position: (" << relativePosition.x << ", " << relativePosition.y << ")" << std::endl;

                // 计算相对位置的模长（距离）
                double distance = sqrt(relativePosition.x * relativePosition.x + relativePosition.y * relativePosition.y);

                if (distance > 0) // 确保距离不为零
                {
                    // 计算当前角度，使用arcsin
                    double sinTheta = relativePosition.y / distance; // 使用y分量与模长的比值
                    double currentAngle = asin(sinTheta);

                    // 确保角度在正确轮限，使用x分量来确定方向
                    if (relativePosition.x < 0)
                    {
                        currentAngle = CV_PI - currentAngle; // 如果x为负，则角度需要在第二或第三轮限
                    }

                    // 如果有上一帧的数据，则计算角速度
                    if (hasPrevData)
                    {
                        // 计算时间差（秒）
                        double deltaTime = (t.count() - prevTime) / 1000.0; // 毫秒转换为秒

                        // 计算角度差，考虑到角度突变的情况，进行平滑处理
                        double deltaAngle = currentAngle - prevAngle;
                        if (deltaAngle > CV_PI)
                        {
                            deltaAngle -= 2 * CV_PI;
                        }
                        else if (deltaAngle < -CV_PI)
                        {
                            deltaAngle += 2 * CV_PI;
                        }

                        // 确保角速度恒为正值
                        double angularVelocity = abs(deltaAngle / deltaTime);
                        observations.emplace_back(t.count() - initialTime, angularVelocity - 1.3);

                        std::cout << "Angular Velocity: " << angularVelocity << " rad/s" << std::endl;

                        // 输出到CSV文件（可注释掉）
                        if (csvFile.is_open())
                        {
                            csvFile << (t.count() - initialTime) << "," << (angularVelocity - 1.3) << "\n";
                        }
                    }

                    // 更新上一帧的数据
                    prevRelativePosition = relativePosition;
                    prevTime = t.count();
                    prevAngle = currentAngle;
                    hasPrevData = true;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
        std::cout << "Recognition Time: " << elapsed_time.count() << " ms" << std::endl;

        imshow("Windmill", src);
        if (waitKey(1) >= 0)
            break;

        frame_count++;
    }

    // 关闭CSV文件（可注释掉）
    if (csvFile.is_open())
    {
        csvFile.close();
    }

    // 调用旋转观测的拟合函数
    fitWindmillRotation(observations);

    return 0;
}