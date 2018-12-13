#pragma once
#ifdef HAVE_CONFIG_H
#include "config_auto.h"
#endif

#include <iostream>

#include "allheaders.h"
#include "baseapi.h"
#include "basedir.h"
#include "dict.h"
#include "openclwrapper.h"
#include "osdetect.h"
#include "renderer.h"
#include "strngs.h"
#include "tprintf.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp" 

using namespace std;
using namespace cv;


//bool detectText(const char* image, std::vector<tesseract::DetectedText> &detectedTextList);
bool detectText(Mat &im, vector<pair<string, RotatedRect>> &outText);