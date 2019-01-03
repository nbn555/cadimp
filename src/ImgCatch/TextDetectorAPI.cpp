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

void setTrainingDataPath(const std::string &str) {
    g_traning_data_path = str;
}

void setMinHeightText(int h)
{
    g_min_height_text = h;
}

bool cropByContour(Mat &src, vector<Point2i> &contour, Mat &cropped, RotatedRect &rect) {
	// rect is the RotatedRect (I got it from a contour...)
	rect = minAreaRect(contour);
	// matrices we'll use
	Mat M, rotated;
	// get angle and size from the bounding box
	cout << rect.angle << endl;
	if ((contour.size() > 4 && rect.size.width < rect.size.height)
		|| (contour.size() == 4 && rect.size.width > rect.size.height)) {
		if (rect.angle == 0) {
			rect.angle = -90;
		}
		else {
			rect.angle += 90.0;
		}
		swap(rect.size.width, rect.size.height);
	}
	Size2f rect_size = rect.size;
	//cout << angle << endl;
	rect_size.width += 2;
	rect_size.height += 2;
	// get the rotation matrix
	M = getRotationMatrix2D(rect.center, rect.angle, 1.0);
	// perform the affine transformation
	warpAffine(src, rotated, M, src.size(), INTER_CUBIC);//INTER_LANCSOZ4
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

void findCharacterRects(Mat& src, vector<RotatedRect> &filteredRects) {
	filteredRects.clear();
	Mat gray, edge, draw;
	//Convert to gray Mat
	cvtColor(src, gray, CV_BGR2GRAY);
	//Filter noise
	Mat blurMat;
	GaussianBlur(gray, blurMat, Size(5, 5), 0);
	//Convert to binary Mat
	Mat binaryMat;
	threshold(gray, binaryMat, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

	
	//Find contours
	vector<vector<Point2i>> contours;
	vector<Vec4i> hierarchy;
	findContours(binaryMat, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	Mat contourMat(binaryMat.size(), CV_8UC1, Scalar::all(0));
	drawContours(contourMat, contours, -1, Scalar::all(255));
	imshow("contourMat", contourMat);
	waitKey(0);
	vector<Vec4i> lines;
	HoughLinesP(contourMat, lines, 1, CV_PI / 180, 200, 50, 10);
	Mat lineMat;
	cvtColor(contourMat, lineMat, CV_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(lineMat, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255));
	}
	//imshow("source", src);
	imshow("detected lines", lineMat);
	imwrite("detectedlines.jpg", lineMat);
	waitKey();
	//Filter contours
	for (size_t i = 0; i < contours.size(); i++)
	{
		/*Mat contourMat = src.clone();
		drawContours(contourMat, contours, i, Scalar(0, 0, 255), 2);*/
		//Skip inside contours
		if (hierarchy[i][3] != -1) {
			continue;
		}
		//Find rotate rectangle of contour
		RotatedRect rect = minAreaRect(contours[i]);
		//For character height > character width, wap width & height of rectangle if width > height
		if (rect.size.width > rect.size.height) {
			swap(rect.size.width, rect.size.height);

			if (rect.angle == 0) {
				rect.angle = -90;
			}
			else {
				rect.angle += 90.0;
			}
		}
		//Skip big rectangle
		if (rect.size.height > src.rows / 10)
		{
			continue;
		}

		// added by Nghi: Skip small rectangle
		if (min(rect.size.width, rect.size.height) < g_min_height_text)
			continue;

		filteredRects.push_back(rect);
	}
}

void groupCharacterRects(vector<RotatedRect> &filteredRects, vector<vector<RotatedRect>> &groupedRects) {
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
				//Don't group rectangles having angles are difference too much
				if (abs(groupedRect->angle - filteredRect->angle) > 45)
				{
					continue;
				}
				//Group the rectangles are quite near together
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
}

void findTextOfLine(Mat &src, vector<vector<RotatedRect>> &groupedRects, tesseract::TessBaseAPI *ocr, vector<pair<string, RotatedRect>> &outText){
	Mat outTextMat = src.clone();
	for (size_t i = 0; i < groupedRects.size(); i++)
	{
		//Mat groupedRectMat = src.clone();
		vector<RotatedRect> aRectGroup = groupedRects[i];
		vector<Point> wordContour;
		for (size_t j = 0; j < aRectGroup.size(); j++)
		{
			//drawRotatedRectangle(groupedRectMat, aRectGroup[j]);
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
	ocr->Init(g_traning_data_path.c_str(), "eng"/*"deu"*/, tesseract::OEM_DEFAULT);
	// Set Page segmentation mode to PSM_AUTO (3)
	ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
	ocr->SetVariable("tessedit_char_whitelist", "φ0123456789abcdefjhijklmnopqrstuvwxyzABCDEFJHIJKLMNOPQRSTUVWXYZ.,+-");


	double scale = 640.0 / src.size().width;
	//resize(src, src, cv::Size(), scale, scale);
	/*namedWindow("Original image", CV_WINDOW_AUTOSIZE);
	imshow("Original image", src);*/
	vector<RotatedRect> filteredRects;
	findCharacterRects(src, filteredRects);
	//group contours
	vector<vector<RotatedRect>> groupedRects;
	groupCharacterRects(filteredRects, groupedRects);

	findTextOfLine(src, groupedRects, ocr, outText);
	delete(ocr);
	return true;
}
