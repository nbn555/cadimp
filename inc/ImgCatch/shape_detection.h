#ifndef ICAD_SHAPEDETECTION_H
#define ICAD_SHAPEDETECTION_H

#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
using namespace cv;
using namespace std;

struct Point2Dd {
    double _v[2];
    Point2Dd() {}
    Point2Dd(double x, double y) {
        _v[0] = x;
        _v[1] = y;
    }
};
struct sline {
    Point2Dd p1;
    Point2Dd p2;
};

struct circle {
	Point center;
	double r;
};

void getLines(Mat src, std::vector<sline>& lines);

#endif