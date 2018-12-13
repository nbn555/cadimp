#pragma once
#ifdef HAVE_CONFIG_H
#include "config_auto.h"
#endif


#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp" 

using namespace std;
using namespace cv;


bool detectText(Mat &im, vector<pair<string, RotatedRect>> &outText);