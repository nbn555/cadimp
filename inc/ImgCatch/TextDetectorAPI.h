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

bool detectText(const char* image, std::vector<tesseract::DetectedText> &detectedTextList);
