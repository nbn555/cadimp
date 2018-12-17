/**********************************************************************
* File:        tessedit.cpp  (Formerly tessedit.c)
* Description: Main program for merge of tess and editor.
* Author:                  Ray Smith
* Created:                 Tue Jan 07 15:21:46 GMT 1992
*
* (C) Copyright 1992, Hewlett-Packard Ltd.
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
** http://www.apache.org/licenses/LICENSE-2.0
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*
**********************************************************************/

// Include automatically generated configuration file if running autoconf

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
#include "TextDetectorAPI.h"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

using namespace std;
using namespace cv;

std::string g_traning_data_path;
int g_min_height_text = 5;
#if defined(HAVE_TIFFIO_H) && defined(_WIN32)

#include <tiffio.h>

static void Win32WarningHandler(const char* module, const char* fmt,
                                va_list ap) {
  if (module != NULL) {
    fprintf(stderr, "%s: ", module);
  }
  fprintf(stderr, "Warning, ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, ".\n");
}

#endif /* HAVE_TIFFIO_H &&  _WIN32 */

void setTrainingDataPath(const std::string &str) {
    g_traning_data_path = str;
}

void setMinHeightText(int h)
{
    g_min_height_text = h;
}

void PrintVersionInfo() {
  char* versionStrP;

  printf("tesseract %s\n", tesseract::TessBaseAPI::Version());

  versionStrP = getLeptonicaVersion();
  printf(" %s\n", versionStrP);
  lept_free(versionStrP);

  versionStrP = getImagelibVersions();
  printf("  %s\n", versionStrP);
  lept_free(versionStrP);

#ifdef USE_OPENCL
  cl_platform_id platform[4];
  cl_uint num_platforms;

  printf(" OpenCL info:\n");
  if (clGetPlatformIDs(4, platform, &num_platforms) == CL_SUCCESS) {
    printf("  Found %u platform(s).\n", num_platforms);
    for (unsigned n = 0; n < num_platforms; n++) {
      char info[256];
      if (clGetPlatformInfo(platform[n], CL_PLATFORM_NAME, 256, info, 0) ==
          CL_SUCCESS) {
        printf("  Platform %u name: %s.\n", n + 1, info);
      }
      if (clGetPlatformInfo(platform[n], CL_PLATFORM_VERSION, 256, info, 0) ==
          CL_SUCCESS) {
        printf("  Version: %s.\n", info);
      }
      cl_device_id devices[2];
      cl_uint num_devices;
      if (clGetDeviceIDs(platform[n], CL_DEVICE_TYPE_ALL, 2, devices,
                         &num_devices) == CL_SUCCESS) {
        printf("  Found %u device(s).\n", num_devices);
        for (unsigned i = 0; i < num_devices; ++i) {
          if (clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 256, info, 0) ==
              CL_SUCCESS) {
            printf("    Device %u name: %s.\n", i + 1, info);
          }
        }
      }
    }
    }
#endif
}

void PrintUsage(const char* program) {
  printf(
      "Usage:\n"
      "  %s --help | --help-psm | --help-oem | --version\n"
      "  %s --list-langs [--tessdata-dir PATH]\n"
      "  %s --print-parameters [options...] [configfile...]\n"
      "  %s imagename|stdin outputbase|stdout [options...] [configfile...]\n",
      program, program, program, program);
}

void PrintHelpForPSM() {
  const char* msg =
      "Page segmentation modes:\n"
      "  0    Orientation and script detection (OSD) only.\n"
      "  1    Automatic page segmentation with OSD.\n"
      "  2    Automatic page segmentation, but no OSD, or OCR.\n"
      "  3    Fully automatic page segmentation, but no OSD. (Default)\n"
      "  4    Assume a single column of text of variable sizes.\n"
      "  5    Assume a single uniform block of vertically aligned text.\n"
      "  6    Assume a single uniform block of text.\n"
      "  7    Treat the image as a single text line.\n"
      "  8    Treat the image as a single word.\n"
      "  9    Treat the image as a single word in a circle.\n"
      " 10    Treat the image as a single character.\n"
      " 11    Sparse text. Find as much text as possible in no"
      " particular order.\n"
      " 12    Sparse text with OSD.\n"
      " 13    Raw line. Treat the image as a single text line,\n"
      "\t\t\tbypassing hacks that are Tesseract-specific.\n";

  printf("%s", msg);
}

void PrintHelpForOEM() {
  const char* msg =
      "OCR Engine modes:\n"
      "  0    Original Tesseract only.\n"
      "  1    Cube only.\n"
      "  2    Tesseract + cube.\n"
      "  3    Default, based on what is available.\n";

  printf("%s", msg);
}

void PrintHelpMessage(const char* program) {
  PrintUsage(program);

  const char* ocr_options =
      "OCR options:\n"
      "  --tessdata-dir PATH   Specify the location of tessdata path.\n"
      "  --user-words PATH     Specify the location of user words file.\n"
      "  --user-patterns PATH  Specify the location of user patterns file.\n"
      "  -l LANG[+LANG]        Specify language(s) used for OCR.\n"
      "  -c VAR=VALUE          Set value for config variables.\n"
      "                        Multiple -c arguments are allowed.\n"
      "  --psm NUM             Specify page segmentation mode.\n"
      "  --oem NUM             Specify OCR Engine mode.\n"
      "NOTE: These options must occur before any configfile.\n";

  printf("\n%s\n", ocr_options);
  PrintHelpForPSM();
  PrintHelpForOEM();

  const char* single_options =
      "Single options:\n"
      "  -h, --help            Show this help message.\n"
      "  --help-psm            Show page segmentation modes.\n"
      "  --help-oem            Show OCR Engine modes.\n"
      "  -v, --version         Show version information.\n"
      "  --list-langs          List available languages for tesseract engine.\n"
      "  --print-parameters    Print tesseract parameters to stdout.\n";

  printf("\n%s", single_options);
}

void SetVariablesFromCLArgs(tesseract::TessBaseAPI* api, int argc,
                            char** argv) {
  char opt1[256], opt2[255];
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
      strncpy(opt1, argv[i + 1], 255);
      opt1[255] = '\0';
      char* p = strchr(opt1, '=');
      if (!p) {
        fprintf(stderr, "Missing = in configvar assignment\n");
        exit(1);
      }
      *p = 0;
      strncpy(opt2, strchr(argv[i + 1], '=') + 1, 255);
      opt2[254] = 0;
      ++i;

      if (!api->SetVariable(opt1, opt2)) {
        fprintf(stderr, "Could not set option: %s=%s\n", opt1, opt2);
      }
    }
  }
}

void PrintLangsList(tesseract::TessBaseAPI* api) {
  GenericVector<STRING> languages;
  api->GetAvailableLanguagesAsVector(&languages);
  printf("List of available languages (%d):\n", languages.size());
  for (int index = 0; index < languages.size(); ++index) {
    STRING& string = languages[index];
    printf("%s\n", string.string());
  }
  api->End();
}

void PrintBanner() {
  tprintf("Tesseract Open Source OCR Engine v%s with Leptonica\n",
          tesseract::TessBaseAPI::Version());
}

/**
 * We have 2 possible sources of pagesegmode: a config file and
 * the command line. For backwards compatibility reasons, the
 * default in tesseract is tesseract::PSM_SINGLE_BLOCK, but the
 * default for this program is tesseract::PSM_AUTO. We will let
 * the config file take priority, so the command-line default
 * can take priority over the tesseract default, so we use the
 * value from the command line only if the retrieved mode
 * is still tesseract::PSM_SINGLE_BLOCK, indicating no change
 * in any config file. Therefore the only way to force
 * tesseract::PSM_SINGLE_BLOCK is from the command line.
 * It would be simpler if we could set the value before Init,
 * but that doesn't work.
 */
void FixPageSegMode(tesseract::TessBaseAPI* api,
                    tesseract::PageSegMode pagesegmode) {
  if (api->GetPageSegMode() == tesseract::PSM_SINGLE_BLOCK)
    api->SetPageSegMode(pagesegmode);
}

// NOTE: arg_i is used here to avoid ugly *i so many times in this function
void ParseArgs(const int argc, char** argv, const char** lang,
               const char** image, const char** outputbase,
               const char** datapath, bool* list_langs, bool* print_parameters,
               GenericVector<STRING>* vars_vec,
               GenericVector<STRING>* vars_values, int* arg_i,
               tesseract::PageSegMode* pagesegmode,
               tesseract::OcrEngineMode* enginemode) {
  if (argc == 1) {
    PrintHelpMessage(argv[0]);
    exit(0);
  }

  if (argc == 2) {
    if ((strcmp(argv[1], "-h") == 0) || (strcmp(argv[1], "--help") == 0)) {
      PrintHelpMessage(argv[0]);
      exit(0);
    }
    if ((strcmp(argv[1], "--help-psm") == 0)) {
      PrintHelpForPSM();
      exit(0);
    }
    if ((strcmp(argv[1], "--help-oem") == 0)) {
      PrintHelpForOEM();
      exit(0);
    }
    if ((strcmp(argv[1], "-v") == 0) || (strcmp(argv[1], "--version") == 0)) {
      PrintVersionInfo();
      exit(0);
    }
  }

  bool noocr = false;
  int i = 1;
  while (i < argc && (*outputbase == NULL || argv[i][0] == '-')) {
    if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
      *lang = argv[i + 1];
      ++i;
    } else if (strcmp(argv[i], "--tessdata-dir") == 0 && i + 1 < argc) {
      *datapath = argv[i + 1];
      ++i;
    } else if (strcmp(argv[i], "--user-words") == 0 && i + 1 < argc) {
      vars_vec->push_back("user_words_file");
      vars_values->push_back(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "--user-patterns") == 0 && i + 1 < argc) {
      vars_vec->push_back("user_patterns_file");
      vars_values->push_back(argv[i + 1]);
      ++i;
    } else if (strcmp(argv[i], "--list-langs") == 0) {
      noocr = true;
      *list_langs = true;
    } else if (strcmp(argv[i], "-psm") == 0 && i + 1 < argc) {
      // The parameter -psm is deprecated and was replaced by --psm.
      // It is still supported for compatibility reasons.
      *pagesegmode = static_cast<tesseract::PageSegMode>(atoi(argv[i + 1]));
      ++i;
    } else if (strcmp(argv[i], "--psm") == 0 && i + 1 < argc) {
      *pagesegmode = static_cast<tesseract::PageSegMode>(atoi(argv[i + 1]));
      ++i;
    } else if (strcmp(argv[i], "--oem") == 0 && i + 1 < argc) {
      *enginemode = static_cast<tesseract::OcrEngineMode>(atoi(argv[i + 1]));
      ++i;
    } else if (strcmp(argv[i], "--print-parameters") == 0) {
      noocr = true;
      *print_parameters = true;
    } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
      // handled properly after api init
      ++i;
    } else if (*image == NULL) {
      *image = argv[i];
    } else if (*outputbase == NULL) {
      *outputbase = argv[i];
    }
    ++i;
  }

  *arg_i = i;

  if (argc == 2 && strcmp(argv[1], "--list-langs") == 0) {
    *list_langs = true;
    noocr = true;
  }

  if (*outputbase == NULL && noocr == false) {
    PrintHelpMessage(argv[0]);
    exit(1);
  }
}

void PreloadRenderers(
    tesseract::TessBaseAPI* api,
    tesseract::PointerVector<tesseract::TessResultRenderer>* renderers,
    tesseract::PageSegMode pagesegmode, const char* outputbase) {
  if (pagesegmode == tesseract::PSM_OSD_ONLY) {
    renderers->push_back(new tesseract::TessOsdRenderer(outputbase));
  } else {
    bool b;
    api->GetBoolVariable("tessedit_create_hocr", &b);
    if (b) {
      bool font_info;
      api->GetBoolVariable("hocr_font_info", &font_info);
      renderers->push_back(
          new tesseract::TessHOcrRenderer(outputbase, font_info));
    }

    api->GetBoolVariable("tessedit_create_tsv", &b);
    if (b) {
      bool font_info;
      api->GetBoolVariable("hocr_font_info", &font_info);
      renderers->push_back(
          new tesseract::TessTsvRenderer(outputbase, font_info));
    }

    api->GetBoolVariable("tessedit_create_pdf", &b);
    if (b) {
      bool textonly;
      api->GetBoolVariable("textonly_pdf", &textonly);
      renderers->push_back(new tesseract::TessPDFRenderer(
          outputbase, api->GetDatapath(), textonly));
    }

    api->GetBoolVariable("tessedit_write_unlv", &b);
    if (b) {
      renderers->push_back(new tesseract::TessUnlvRenderer(outputbase));
    }

    api->GetBoolVariable("tessedit_create_boxfile", &b);
    if (b) {
      renderers->push_back(new tesseract::TessBoxTextRenderer(outputbase));
    }

    api->GetBoolVariable("tessedit_create_txt", &b);
    if (b || renderers->empty()) {
      renderers->push_back(new tesseract::TessTextRenderer(outputbase));
    }
  }

  if (!renderers->empty()) {
    // Since the PointerVector auto-deletes, null-out the renderers that are
    // added to the root, and leave the root in the vector.
    for (int r = 1; r < renderers->size(); ++r) {
      (*renderers)[0]->insert((*renderers)[r]);
      (*renderers)[r] = NULL;
    }
  }
}

bool cropByContour(Mat &src, vector<Point2i> &contour, Mat &cropped, RotatedRect &rect) {
	// rect is the RotatedRect (I got it from a contour...)
	rect = minAreaRect(contour);
	// matrices we'll use
	Mat M, rotated;
	// get angle and size from the bounding box
	Size rect_size = rect.size;
	cout << rect.angle << endl;
	if ((contour.size() > 4 && rect_size.width < rect_size.height)
		|| (contour.size() == 4 && rect_size.width > rect_size.height)) {
		if (rect.angle == 0) {
			rect.angle = -90;
		}
		else {
			rect.angle += 90.0;
		}
		swap(rect_size.width, rect_size.height);
	}

	//cout << angle << endl;
	rect_size.width += 2;
	rect_size.height += 2;
	// get the rotation matrix
	M = getRotationMatrix2D(rect.center, rect.angle, 1.0);
	// perform the affine transformation
	warpAffine(src, rotated, M, src.size(), INTER_CUBIC);
	// crop the resulting image
	getRectSubPix(rotated, rect_size, rect.center, cropped);
	Mat extendMat(cropped.rows * 2, cropped.cols * 2, CV_8UC3, Scalar::all(255));
	if (cropped.rows == 0 || cropped.cols == 0)
	{
		return false;
	}
	cropped.copyTo(extendMat(Rect(cropped.cols / 2, cropped.rows / 2, cropped.cols, cropped.rows)));
	cropped = extendMat.clone();
	return true;
}

void drawRotatedRectangle(cv::Mat& image, RotatedRect &rotatedRectangle)
{
	cv::Scalar color = cv::Scalar(0, 0, 255); // white

											  // We take the edges that OpenCV calculated for us
	cv::Point2f vertices2f[4];
	rotatedRectangle.points(vertices2f);

	// Convert them so we can use them in a fillConvexPoly
	cv::Point vertices[4];
	for (int i = 0; i < 4; ++i) {
		vertices[i] = vertices2f[i];
	}

	// Now we can fill the rotated rectangle with our specified color
	cv::fillConvexPoly(image,
		vertices,
		4,
		color);
}

/**********************************************************************
 *  main()
 *
 **********************************************************************/
//
bool detectText(Mat &src, vector<pair<string, RotatedRect>> &outText) {
	// Create Tesseract object
	tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
	// Initialize tesseract to use English (eng) and the LSTM OCR engine. 
	ocr->Init(g_traning_data_path.c_str(), "eng", tesseract::OEM_DEFAULT);
	// Set Page segmentation mode to PSM_AUTO (3)
	ocr->SetPageSegMode(tesseract::PSM_SINGLE_WORD);
	ocr->SetVariable("tessedit_char_whitelist", "0123456789abcdefjhijklmnopqrstuvwxyzABCDEFJHIJKLMNOPQRSTUVWXYZ.,+-");


	double scale = 640.0 / src.size().width;
	//resize(src, src, cv::Size(), scale, scale);
	/*namedWindow("Original image", CV_WINDOW_AUTOSIZE);
	imshow("Original image", src);*/
	Mat gray, edge, draw;
	cvtColor(src, gray, CV_BGR2GRAY);
	Mat blurMat;
	GaussianBlur(gray, blurMat, Size(5, 5), 0);
	Mat binaryMat;
	threshold(gray, binaryMat, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
	vector<vector<Point2i>> contours;
	vector<Vec4i> hierarchy;
	findContours(binaryMat, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	vector<RotatedRect> filteredRects;
	for (size_t i = 0; i < contours.size(); i++)
	{
		Mat contourMat = src.clone();
		drawContours(contourMat, contours, i, Scalar(0, 0, 255), 2);
		if (hierarchy[i][3] != -1) {
			continue;
		}
		RotatedRect rect = minAreaRect(contours[i]);
		if (rect.size.width > rect.size.height) {
			swap(rect.size.width, rect.size.height);

			if (rect.angle == 0) {
				rect.angle = -90;
			}
			else {
				rect.angle += 90.0;
			}
		}
		if (rect.size.height > src.rows / 10)
		{
			continue;
		}

        // added by Nghi
        if(min(rect.size.width,rect.size.height) < g_min_height_text)
            continue;

		filteredRects.push_back(rect);
	}
	//group contours
	vector<vector<RotatedRect>> groupedRects;
	while (!filteredRects.empty())
	{
		vector<RotatedRect> aRectGroup;
		aRectGroup.push_back(filteredRects.back());
		filteredRects.pop_back();
		RotatedRect *groupedRect, *filteredRect;

		for (size_t i = 0; i < filteredRects.size(); i++)
		{
			filteredRect = &filteredRects[i];
			for (size_t j = 0; j < aRectGroup.size(); j++)
			{
				groupedRect = &aRectGroup[j];
				if (abs(groupedRect->angle - filteredRect->angle) > 45)
				{
					continue;
				}

				if (norm(groupedRect->center - filteredRect->center) < (groupedRect->size.width + filteredRect->size.width) * 2)
				{
					aRectGroup.push_back(*filteredRect);
					filteredRects.erase(filteredRects.begin() + i);
					i--;
					break;
				}
			}
		}
		groupedRects.push_back(aRectGroup);
	}
	Mat outTextMat = src.clone();
	for (size_t i = 0; i < groupedRects.size(); i++)
	{
		Mat groupedRectMat = src.clone();
		vector<RotatedRect> aRectGroup = groupedRects[i];
		vector<Point> wordContour;
		for (size_t j = 0; j < aRectGroup.size(); j++)
		{
			drawRotatedRectangle(groupedRectMat, aRectGroup[j]);
			cv::Point2f vertices2f[4];
			aRectGroup[j].points(vertices2f);

			// Convert them so we can use them in a fillConvexPoly
			for (int i = 0; i < 4; ++i) {
				wordContour.push_back((Point)vertices2f[i]);
			}
		}


		//imshow("groupedRectMat", groupedRectMat);
		Mat cropped;
		RotatedRect rect;
		bool flag = cropByContour(src, wordContour, cropped, rect);
		if (!flag)
		{
			continue;
		}
		//imshow("cropped", cropped);
		string outTextStr;
		
		// Set image data
		ocr->SetImage(cropped.data, cropped.cols, cropped.rows, 3, cropped.step);

		// Run Tesseract OCR on image
		outTextStr = string(ocr->GetUTF8Text());
		if (outTextStr.empty())
		{
			continue;
		}
		// print recognized text
		cout << outTextStr << endl;

		putText(outTextMat, outTextStr, wordContour[0], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
		outText.push_back(pair<string, RotatedRect>(outTextStr, rect));

		//waitKey(0);
	}

	imwrite("outTextMat.jpg", outTextMat);
	return true;
}
