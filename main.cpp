#include "windmill.hpp"
#include "ceres/ceres.h"
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

struct WindmillCost
{
    WindmillCost(double t, double a) : t_(t), a_(a) {}

    template <typename T>
    bool operator()(const T *const p, T *residual) const
    {
        T A = p[0];
        T w = p[1];
        T phi = p[2];
        T b = p[3];
        residual[0] = a_ - cos(A / w * (cos(phi + M_PI / 2) - cos(w * t_ + phi + M_PI / 2)) + b * t_);
        return true;
    }

private:
    const double t_;
    const double a_;
};

void processContours(const vector<vector<Point>> &contours, const vector<Vec4i> &hierarchy, int &flagR, int &flagFan)
{
    const double minAreaR = 500;
    const double minAreaFan = 10000;

    for (size_t i = 0; i < contours.size(); i++)
    {
        if (flagR != -1 && flagFan != -1)
            break;
        if (hierarchy[i][3] == -1)
        {
            if (contourArea(contours[i]) < minAreaR)
            {
                flagR = i;
            }
            else if (contourArea(contours[i]) < minAreaFan)
            {
                flagFan = hierarchy[i][2];
            }
        }
    }
}

int main()
{
    const double threshBinary = 30;
    const double maxBinary = 255;
    const double circleRadius = 2;
    const Scalar circleColor(0, 255, 0);
    const Scalar boxColor(255, 0, 0);
    const double fontScale = 0.5;
    const int fontThickness = 1;
    const double initA = 1.785;
    const double initW = 0.884;
    const double initPhi = 1.24;
    const double initB = 0.305;
    const double lowerBoundW = 0.5;
    const double upperBoundW = 1.90;
    const double lowerBoundPhi = 0.25;
    const double upperBoundPhi = 1.30;
    const double lowerBoundB = 0.5;
    const double upperBoundB = 1.35;
    const int maxIterations = 25;
    const double funcTolerance = 1e-6;
    const double paramTolerance = 1e-6;
    const double condMinA = 0.745;
    const double condMaxA = 0.825;
    const double condMinW = 1.790;
    const double condMaxW = 1.980;
    const double condMinPhi = 0.225;
    const double condMaxPhi = 0.255;
    const double condMinB = 1.240;
    const double condMaxB = 1.370;

    double totalTime = 0;
    for (int trial = 1; trial <= 10; trial++)
    {
        int64 startTime = getTickCount(), endTime;
        auto tStartMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        WINDMILL::WindMill windmillSim(static_cast<double>(tStartMs));

        ceres::Problem optProblem;

        while (true)
        {
            auto tNowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            Mat windmillFrame = windmillSim.getMat(static_cast<double>(tNowMs));
            double elapsedTime = (static_cast<double>(tNowMs) - tStartMs) / 1000;

            Mat grayFrame;
            cvtColor(windmillFrame, grayFrame, COLOR_BGR2GRAY);
            Mat binaryFrame;
            threshold(grayFrame, binaryFrame, threshBinary, maxBinary, THRESH_BINARY);

            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(binaryFrame, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            int outerIdx = -1, FanIdx = -1;
            processContours(contours, hierarchy, outerIdx, FanIdx);

            if (outerIdx == -1 || FanIdx == -1)
                continue;

            Moments outerMoments = moments(contours[outerIdx]);
            Moments FanMoments = moments(contours[FanIdx]);
            Point2d centerOuter(static_cast<int>(outerMoments.m10 / outerMoments.m00), static_cast<int>(outerMoments.m01 / outerMoments.m00));
            Point2d centerFan(static_cast<int>(FanMoments.m10 / FanMoments.m00), static_cast<int>(FanMoments.m01 / FanMoments.m00));
            circle(windmillFrame, centerOuter, circleRadius, circleColor, -1);
            putText(windmillFrame, "Center R", centerOuter, FONT_HERSHEY_SIMPLEX, fontScale, circleColor, fontThickness);
            circle(windmillFrame, centerFan, circleRadius, circleColor, -1);
            putText(windmillFrame, "Fan Center", centerFan, FONT_HERSHEY_SIMPLEX, fontScale, circleColor, fontThickness);

            // Draw edges of fan
            drawContours(windmillFrame, contours, FanIdx, Scalar(255, 0, 0), 2);

            //========================== Parameter Fitting ========================//
            double params[4] = {initA, initW, initPhi, initB};
            double angle = (centerFan.x - centerOuter.x) / norm(centerFan - centerOuter);

            optProblem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<WindmillCost, 1, 4>(
                    new WindmillCost(elapsedTime, angle)),
                nullptr,
                params);

            // Set parameter bounds
            optProblem.SetParameterLowerBound(params, 1, lowerBoundW);
            optProblem.SetParameterUpperBound(params, 1, upperBoundW);
            optProblem.SetParameterLowerBound(params, 2, lowerBoundPhi);
            optProblem.SetParameterUpperBound(params, 2, upperBoundPhi);
            optProblem.SetParameterLowerBound(params, 3, lowerBoundB);
            optProblem.SetParameterUpperBound(params, 3, upperBoundB);

            // Configure the solver
            ceres::Solver::Options solverOptions;
            solverOptions.linear_solver_type = ceres::DENSE_SCHUR;  // Use a faster linear solver
            solverOptions.max_num_iterations = maxIterations;
            solverOptions.function_tolerance = funcTolerance;
            solverOptions.parameter_tolerance = paramTolerance;
            solverOptions.minimizer_progress_to_stdout = false;  // Disable output for speed
            solverOptions.num_threads = 4;  // Use multiple threads if available
            solverOptions.use_nonmonotonic_steps = true;  // Allow non-monotonic steps to converge faster
            solverOptions.preconditioner_type = ceres::JACOBI;  // Use preconditioner to speed up convergence

            ceres::Solver::Summary solverSummary;
            ceres::Solve(solverOptions, &optProblem, &solverSummary);

            // Output the fitting results
            cout << solverSummary.BriefReport() << "\n";
            cout << "Estimated Parameters: " << endl;
            cout << "A: " << params[0] << endl;
            cout << "w: " << params[1] << endl;
            cout << "phi: " << params[2] << endl;
            cout << "b: " << params[3] << endl;

            // Display the processed frame
            imshow("Windmill Detection", windmillFrame);
            if (waitKey(1) >= 0)
                break;

            if (condMinA < params[0] && params[0] < condMaxA &&
                condMinW < params[1] && params[1] < condMaxW &&
                condMinPhi < params[2] && params[2] < condMaxPhi &&
                condMinB < params[3] && params[3] < condMaxB)
            {
                endTime = getTickCount();
                totalTime += (endTime - startTime) / getTickFrequency();
                break;
            }
        }
    }
    cout << "Average Time: " << totalTime / 10 << " seconds" << endl;
    return 0;
}