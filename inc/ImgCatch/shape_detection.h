#ifndef ICAD_SHAPEDETECTION_H
#define ICAD_SHAPEDETECTION_H

#include "opencv2\core\core.hpp"
#include <vector>

void detectLines(cv::Mat src, std::vector<cv::Vec4i> &lines);
void detectArrows(const cv::Mat &src, std::vector<std::vector<cv::Point2f>> &vtArrows,
    const double &standard_area = 400.0, const double &standard_ratio = 0.5);
void getIntersections(cv::Mat src, std::vector<cv::Point> &vtIntersectionPoints, int flag = 0);

#endif
