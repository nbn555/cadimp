#pragma once
#ifdef HAVE_CONFIG_H
#include "config_auto.h"
#endif


#include "opencv2\core\core.hpp"


//extern std::string g_traning_data_path;
void setMinHeightText(int h);
void setTrainingDataPath(const std::string &str);
bool detectText(cv::Mat &im, std::vector<std::pair<std::string, cv::RotatedRect>> &outText, std::string path = "");