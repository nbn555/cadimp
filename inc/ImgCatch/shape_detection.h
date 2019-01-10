#ifndef ICAD_SHAPEDETECTION_H
#define ICAD_SHAPEDETECTION_H

#include "opencv2\core\core.hpp"
#include <vector>

void detectLines(cv::Mat src, std::vector<cv::Vec4i> &lines);
void detectArrows(cv::Mat, std::vector<std::vector<cv::Point2f>>&, const double&, const double&);
void getIntersections(cv::Mat src, std::vector<cv::Point> &vtIntersectionPoints, int flag = 0);

#endif
