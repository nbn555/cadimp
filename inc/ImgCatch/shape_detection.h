#ifndef ICAD_SHAPEDETECTION_H
#define ICAD_SHAPEDETECTION_H

#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
using namespace cv;
using namespace std;

struct Point2dd {
    union
    {
        double _v[2];
        struct
        {
            double x;
            double y;
        };
    };
    Point2dd() {}
    Point2dd(double x, double y) {
        _v[0] = x;
        _v[1] = y;
    }
};
struct sline {
    Point2dd p1;
    Point2dd p2;
};

struct circle {
	Point center;
	double r;
};

void getLines(Mat src, std::vector<sline>& lines);

#endif