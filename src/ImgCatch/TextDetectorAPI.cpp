﻿/**********************************************************************
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
#include <string>
#include <locale>
#include <codecvt>
#include "dirent.h"
#include "shape_detection.h"
#include "LiProcess.h"
#include <algorithm>

#define SHOW_DEBUG

using namespace std;
using namespace cv;

std::string g_traning_data_path;
int g_min_height_text = 5;
int g_min_size_text = 15;

void setTrainingDataPath(const std::string &str) {
    g_traning_data_path = str;
}

void setMinHeightText(int h)
{
    g_min_height_text = h;
}

void detectTextOfFolder(const std::string &path) {
	DIR *pDIR;
	struct dirent *entry;
	//Mat original_mat;
	std::string full_path;
	if (pDIR = opendir(path.c_str())) {
		while (entry = readdir(pDIR)) {
			if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
				std::string newPath = path + "\\" + entry->d_name;
				if (entry->d_type == DT_DIR) {
					detectTextOfFolder(newPath);
					continue;
				}
				Mat src, src_gray;

				/// Read the image
				src = imread(newPath);
				if (!src.data)
				{
					continue;
				}
				Mat removedBorderMat;
				removeBorder(src, removedBorderMat);
				vector<pair<std::string, RotatedRect>> outText;
				detectText(removedBorderMat, outText, newPath);
			}
		}
		closedir(pDIR);
		delete entry;
	}
}

void drawRotatedRectangle(cv::Mat& image, RotatedRect &rotatedRectangle, Scalar color = Scalar(0, 0, 255))
{
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

bool correctRect(Rect &rect, Size size) {
	if (rect.x >= size.width - 5 || rect.y >= size.height - 5)
	{
		return false;
	}
	if (rect.x + rect.width > size.width - 1)
	{
		rect.width = size.width - rect.x - 1;
	}
	if (rect.y + rect.height > size.height - 1)
	{
		rect.height = size.height - rect.y - 1;
	}
}

bool cropByContour(Mat &src, vector<RotatedRect>& aRectGroup, Mat &cropped, RotatedRect& boundingRect) {
	vector<Point> wordContour;
	vector<Point> centerPoints;
	for (size_t j = 0; j < aRectGroup.size(); j++)
	{
		cv::Point2f vertices2f[4];
		aRectGroup[j].points(vertices2f);
		for (int i = 0; i < 4; ++i) {
			wordContour.push_back((Point)vertices2f[i]);
		}

		centerPoints.push_back(aRectGroup[j].center);
	}

	boundingRect = minAreaRect(wordContour);
	/*Mat groupedRectMat = src.clone();
	drawRotatedRectangle(groupedRectMat, boundingRect);
	namedWindow("rectangle", CV_WINDOW_NORMAL);
	setWindowProperty("rectangle", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	imshow("rectangle", groupedRectMat);
	waitKey(0);*/
	float angle;
	if (aRectGroup.size() == 1) {
		angle = boundingRect.angle;
		if (boundingRect.size.width > boundingRect.size.height)
		{
			angle = boundingRect.angle > -2.0 && boundingRect.angle <= 0.0 ? -90.0 : angle + 90.0;
			swap(boundingRect.size.width, boundingRect.size.height);
		}
	}
	else {
		if (aRectGroup.size() == 2)
		{
			Point pointDiff = centerPoints[1] - centerPoints[0];
			angle = cvFastArctan(pointDiff.y, pointDiff.x);
			if (angle > 90 & angle <= 180)
			{
				angle -= 180;
			}
			else if (angle > 180 && angle <= 270)
			{
				angle -= 180;
			}
			else if (angle > 270 && angle <= 360)
			{
				angle -= 360;
			}
			float angleDistance = abs(boundingRect.angle - angle);
			if (angleDistance < 45 || angleDistance > 135) {
				angle = boundingRect.angle;
			}
			else
			{
				angle = boundingRect.angle > -2.0 && boundingRect.angle <= 0.0 ? -90.0 : boundingRect.angle + 90.0;
				swap(boundingRect.size.width, boundingRect.size.height);
			}
		}
		else
		{
			RotatedRect centerRect = minAreaRect(centerPoints);
			angle = centerRect.angle;
			if (centerRect.size.width < centerRect.size.height)
			{
				angle = angle > -2.0 && angle <= 0.0 ? -90.0 : angle + 90.0;
			}
			float angleDistance = abs(boundingRect.angle - angle);
			if (angleDistance > 45 && angleDistance < 135)
			{
				swap(boundingRect.size.width, boundingRect.size.height);
			}
		}
	}
	boundingRect.angle = angle;
	
	Size2f rect_size = boundingRect.size;
	//cout << angle << endl;
	rect_size.width += 2;
	rect_size.height += 2;
	// get the rotation matrix
	Mat M, rotated;
	int maxRectSize = max(rect_size.width, rect_size.height) * 2.5;
	cv::Rect rect(boundingRect.center - Point2f(maxRectSize / 2, maxRectSize / 2), Size2i(maxRectSize, maxRectSize));
	correctRect(rect, src.size());
	cropped = src(rect);
	M = getRotationMatrix2D(Point2f(cropped.cols/2, cropped.rows/2), boundingRect.angle, 1.0);
	// perform the affine transformation
	imshow("cropped", cropped);
	warpAffine(cropped, rotated, M, cropped.size(), INTER_CUBIC);//INTER_LANCSOZ4
	imshow("rotated", rotated);
	// crop the resulting image
	//Point2f croppedRectCenter = boundingRect.center + Point2f(maxRectSize, maxRectSize);
	getRectSubPix(rotated, rect_size, Point2f(rotated.cols/2, rotated.rows/2),cropped);
	imshow("cropped1", cropped);
	Mat extendMat(cropped.rows * 2.5, cropped.cols * 2.5, CV_8UC3, Scalar::all(255));
	if (cropped.rows == 0 || cropped.cols == 0)
	{
		return false;
	}
	rect = Rect(cropped.cols / 2, cropped.rows / 2, cropped.cols, cropped.rows);
	//correctRect(rect, cropped.size());
	cropped.copyTo(extendMat(rect));
	cropped = extendMat.clone();
	/*imshow("cropped2", cropped);
	waitKey();*/

	return true;
}

void findCharacterRects(Mat& src, vector<RotatedRect> &filteredRects, std::string path /*= ""*/) {
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
	findContours(binaryMat.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	Mat contourMat(binaryMat.size(), CV_8UC1, Scalar::all(0));
	drawContours(contourMat, contours, -1, Scalar::all(255));
	/*imshow("contourMat", contourMat);
	waitKey(0);*/
	vector<Vec4i> lines;
	HoughLinesP(contourMat, lines, 1, CV_PI / 180, 200, 100, 10);
	Mat lineMat;
	cvtColor(contourMat, lineMat, CV_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(lineMat, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255));
		line(binaryMat, Point(l[0], l[1]), Point(l[2], l[3]), Scalar::all(0), 9);
	}

	for (size_t i = 0; i < contours.size(); i++)
	{
		RotatedRect rect = minAreaRect(contours[i]);
		if (rect.size.height > src.rows / 10) {
			drawContours(binaryMat, contours, i, Scalar::all(0), 9);
		}
	}
	//imshow("source", src);
	//imshow("detected lines", lineMat);
	std::string newPath = path + "Detectedlines.jpg";
	imwrite(newPath, binaryMat);
	//waitKey();
	findContours(binaryMat.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
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
		if (min(rect.size.width, rect.size.height) < g_min_height_text
			|| max(rect.size.width, rect.size.height) < g_min_size_text)
			continue;

		filteredRects.push_back(rect);
	}
}

void findNearRects(std::vector<RotatedRect> &rects, std::vector<RotatedRect> &oldNearRects, std::vector<RotatedRect> &newNearRects, Mat &src) {
	newNearRects.clear();
	RotatedRect *aRect, *rect;
	for (size_t oldId = 0; oldId < oldNearRects.size(); oldId++)
	{
		rect = &oldNearRects[oldId];
		for (size_t rectId = 0; rectId < rects.size(); rectId++)
		{
			aRect = &rects[rectId];
#ifdef SHOW_DEBUG_
			Mat rectMat1 = src.clone();
			drawRotatedRectangle(rectMat1, *rect, Scalar(0, 0, 255));
			drawRotatedRectangle(rectMat1, *aRect, Scalar(0, 255, 255));
			namedWindow("rectangle", CV_WINDOW_NORMAL);
			setWindowProperty("rectangle", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			imshow("rectangle", rectMat1);
			//waitKey(0);
			int ch = waitKeyEx(0);
			if (ch == '1')
			{
				break;
			}
#endif // SHOW_DEBUG
			//Don't group rectangles having angles are difference too much
			if (abs(rect->angle - aRect->angle) > 65 && abs(abs(rect->angle - aRect->angle) - 180) > 10)
			{
				continue;
			}
			//Group the rectangles are quite near together
			double distance = norm(rect->center - aRect->center);
			if (distance < 50 || distance < min(rect->size.height,aRect->size.height) * 2)
			{
#ifdef SHOW_DEBUG
				drawRotatedRectangle(src, *aRect);
#endif // SHOW_DEBUG
				newNearRects.push_back(*aRect);
				rects.erase(rects.begin() + rectId);
				rectId--;
			}
		}
	}
}

void findARectGroup(std::vector<RotatedRect> &rects, std::vector<RotatedRect> &groupedRects, Mat &src) {
	groupedRects.clear();
	RotatedRect rect = rects.back();
#ifdef SHOW_DEBUG
		Mat rectMat = src.clone();
		drawRotatedRectangle(rectMat, rect);
#endif // SHOW_DEBUG
	rects.pop_back();
	std::vector<RotatedRect> oldRectGroup, newRectGroup;
	oldRectGroup.push_back(rect);
	groupedRects.push_back(rect);
	do
	{
		findNearRects(rects, oldRectGroup, newRectGroup, src.clone());
		oldRectGroup.clear();
		oldRectGroup.assign(newRectGroup.begin(), newRectGroup.end());
		for (size_t i = 0; i < newRectGroup.size(); i++)
		{
			groupedRects.push_back(newRectGroup[i]);
		}
	} while (!newRectGroup.empty() && !rects.empty());
}

void groupCharacterRects(vector<RotatedRect> &filteredRects, vector<vector<RotatedRect>> &groupedRects, Mat& src) {
	while (!filteredRects.empty())
	{
		vector<RotatedRect> aRectGroup;
		findARectGroup(filteredRects, aRectGroup, src);
		groupedRects.push_back(aRectGroup);
	}
}
#if 0
void findTextOfLine(Mat &src, vector<vector<RotatedRect>> &groupedRects, tesseract::TessBaseAPI *ocr, vector<pair<string, RotatedRect>> &outText, std::string path = ""){
	Mat outTextMat = src.clone();
	for (size_t i = 0; i < groupedRects.size(); i++)
	{
		Mat groupedRectMat = src.clone();
		vector<RotatedRect> aRectGroup = groupedRects[i];
		vector<Point> wordContour;
		float angleTotal = 0;
		float angle;
		for (size_t j = 0; j < aRectGroup.size(); j++)
		{
			angle = aRectGroup[j].angle;
			if (aRectGroup[j].size.width > aRectGroup[j].size.height) {
				if (aRectGroup[j].angle == 0) {
					angle = -90;
				}
				else {
					angle += 90.0;
				}
			}
			angleTotal += angle;
			drawRotatedRectangle(groupedRectMat, aRectGroup[j]);
			cv::Point2f vertices2f[4];
			aRectGroup[j].points(vertices2f);

			// Convert them so we can use them in a fillConvexPoly
			for (int i = 0; i < 4; ++i) {
				wordContour.push_back((Point)vertices2f[i]);
			}
		}

		angle = angleTotal / aRectGroup.size();
		//namedWindow("rectangle", CV_WINDOW_NORMAL);
		//setWindowProperty("rectangle", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		//imshow("rectangle", groupedRectMat);
		//waitKey(0);
		Mat cropped;
		RotatedRect rect;
		bool flag = cropByContour(src, wordContour, cropped, rect, angle);
		if (!flag)
		{
			continue;
		}
		/*imshow("cropped", cropped);
		waitKey(0);*/
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

		outTextStr.erase(std::remove(outTextStr.begin(), outTextStr.end(), '\n'), outTextStr.end());
		putText(outTextMat, outTextStr, wordContour[0], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
		outText.push_back(pair<string, RotatedRect>(outTextStr, rect));

		//waitKey(0);
	}
	string newPath = path + "TextMat.jpg";
	imwrite(newPath, outTextMat);

}

#endif
std::wstring utf8_to_utf16(const std::string& utf8)
{
	std::vector<unsigned long> unicode;
	size_t i = 0;
	while (i < utf8.size())
	{
		unsigned long uni;
		size_t todo;
		bool error = false;
		unsigned char ch = utf8[i++];
		if (ch <= 0x7F)
		{
			uni = ch;
			todo = 0;
		}
		else if (ch <= 0xBF)
		{
			throw std::logic_error("not a UTF-8 string");
		}
		else if (ch <= 0xDF)
		{
			uni = ch & 0x1F;
			todo = 1;
		}
		else if (ch <= 0xEF)
		{
			uni = ch & 0x0F;
			todo = 2;
		}
		else if (ch <= 0xF7)
		{
			uni = ch & 0x07;
			todo = 3;
		}
		else
		{
			throw std::logic_error("not a UTF-8 string");
		}
		for (size_t j = 0; j < todo; ++j)
		{
			if (i == utf8.size())
				throw std::logic_error("not a UTF-8 string");
			unsigned char ch = utf8[i++];
			if (ch < 0x80 || ch > 0xBF)
				throw std::logic_error("not a UTF-8 string");
			uni <<= 6;
			uni += ch & 0x3F;
		}
		if (uni >= 0xD800 && uni <= 0xDFFF)
			throw std::logic_error("not a UTF-8 string");
		if (uni > 0x10FFFF)
			throw std::logic_error("not a UTF-8 string");
		unicode.push_back(uni);
	}
	std::wstring utf16;
	for (size_t i = 0; i < unicode.size(); ++i)
	{
		unsigned long uni = unicode[i];
		if (uni <= 0xFFFF)
		{
			utf16 += (wchar_t)uni;
		}
		else
		{
			uni -= 0x10000;
			utf16 += (wchar_t)((uni >> 10) + 0xD800);
			utf16 += (wchar_t)((uni & 0x3FF) + 0xDC00);
		}
	}
	return utf16;
}

void replaceAll(std::wstring& str, const std::wstring& from, const std::wstring& to) {
	if (from.empty())
		return;
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::wstring::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
	}
}

void splitLines(char* str, std::vector<std::string> &splittedStrs)
{
	// Returns first token  
	char *token = strtok(str, "\n");

	// Keep printing tokens while one of the 
	// delimiters present in str[]. 
	while (token != NULL)
	{
		std::string subStr(token);
		if (!subStr.empty())
		{
			splittedStrs.push_back(std::string(token));
		}
		token = strtok(NULL, " ");
	}
}

void replaceChar(string& str) {
	str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
	std::wstring sLogLevel = utf8_to_utf16(str);
	//outTextStr.replace(outTextStr.begin(), outTextStr.end(), '¢', 'a');
	replaceAll(sLogLevel, L"¢", L"phi");
	replaceAll(sLogLevel, L"@", L"phi");
	str = std::string(sLogLevel.begin(), sLogLevel.end());
}

void findTextOfLine(Mat &src, vector<vector<RotatedRect>> &groupedRects, tesseract::TessBaseAPI *ocr, vector<pair<string, RotatedRect>> &outText, std::string path/* = ""*/){
    Mat outTextMat = src.clone();
   // Mat outTextMatShow = src.clone();
	for (size_t i = 0; i < groupedRects.size(); i++)
	{
		//Mat groupedRectMat = src.clone();
		vector<RotatedRect> aRectGroup = groupedRects[i];
		//vector<Point> wordContour;
		//vector<Point> centerPoints;
		////float angleTotal = 0;
		////float angle;
		//for (size_t j = 0; j < aRectGroup.size(); j++)
		//{
		//	/*angle = aRectGroup[j].angle;
		//	if (aRectGroup[j].size.width > aRectGroup[j].size.height) {
		//		if (aRectGroup[j].angle == 0) {
		//			angle = -90;
		//		}
		//		else {
		//			angle += 90.0;
		//		}
		//	}
		//	angleTotal += angle;*/
		//	//drawRotatedRectangle(groupedRectMat, aRectGroup[j]);
		//	cv::Point2f vertices2f[4];
		//	aRectGroup[j].points(vertices2f);
		//	for (int i = 0; i < 4; ++i) {
		//		wordContour.push_back((Point)vertices2f[i]);
		//	}

		//	centerPoints.push_back(aRectGroup[j].center);
		//}

//		angle = angleTotal / aRectGroup.size();
        float rangle;
        if (centerPs.size() > 1) {
            float dx = centerPs[0].x - centerPs.back().x;
            float dy = centerPs[0].y - centerPs.back().y;
            if (abs(dx) > abs(dy)) {
                std::sort(centerPs.begin(), centerPs.end(),
                    [=](const Point &pl, const Point &pr)->bool {
                    return pl.x < pr.x; });
                dx = abs(centerPs[0].x - centerPs.back().x);
                dy = abs(centerPs[0].y - centerPs.back().y);
                float l = sqrt(dx*dx + dy*dy);
                rangle = atan2(dy / l, dx / l);
                rangle *= 180 / 3.1415926;
            //    angle = rangle;
            }
            else {
                std::sort(centerPs.begin(), centerPs.end(),
                    [=](const Point &pl, const Point &pr)->bool {
                    return pl.y < pr.y; });
                dx = abs(centerPs[0].x - centerPs.back().x);
                dy = abs(centerPs[0].y - centerPs.back().y);
                float l = sqrt(dx*dx + dy*dy);
                rangle = atan2(dy / l, dx / l);
                rangle *= 180 / 3.1415926;
         //       angle = rangle;
            }
        }
		/*namedWindow("rectangle", CV_WINDOW_NORMAL);
		setWindowProperty("rectangle", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		imshow("rectangle", groupedRectMat);*/
		//waitKey(0);
		Mat cropped;
		RotatedRect rect;
		bool flag = cropByContour(src, aRectGroup, cropped, rect);
		if (!flag)
		{
			continue;
		}

       // drawRotatedRectangle(outTextMatShow, rect);
#ifdef SHOW_DEBUG
		/*imshow("cropped", cropped);
		waitKey(0);*/
		string croppedName = path + std::to_string(i) + ".jpg";
		imwrite(croppedName, cropped);
#endif
		// Set image data
		/*if ()
		{
			ocr->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
		}
		else*/ if (aRectGroup.size() == 1 || cropped.cols > cropped.rows){
			ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
		}
		else {
			ocr->SetPageSegMode(tesseract::PSM_SPARSE_TEXT);
		}
		ocr->SetImage(cropped.data, cropped.cols, cropped.rows, 3, cropped.step);

		// Run Tesseract OCR on image
		char* outTextCh = ocr->GetUTF8Text();
		if (!outTextCh)
		{
			continue;
		}
		std::vector<std::string> textLines;
		splitLines(outTextCh, textLines);
		//outTextStr.erase(std::remove(outTextStr.begin(), outTextStr.end(), '\n'), outTextStr.end());
		// print recognized text
		//cout << outTextStr << endl;
		for (int textLineId = 0; textLineId < textLines.size(); textLineId++)
		{
			textLines[textLineId].erase(std::remove(textLines[textLineId].begin(), textLines[textLineId].end(), '\n'), textLines[textLineId].end());
			if (textLines[textLineId].empty())
			{
				textLines.erase(textLines.begin() + textLineId);
				textLineId--;
			}
		}
		if (textLines.empty())
		{
			continue;
		}
		Point2f points[4];
		rect.points(points);
		if (textLines.size() == 1)
		{
			string outTextStr = outTextCh;
			replaceChar(outTextStr);
			putText(outTextMat, outTextStr, rect.center, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
			outText.push_back(pair<string, RotatedRect>(outTextStr, rect));
			continue;
		}
		
		Point2f rectPoints[4];
		rect.points(rectPoints);
		RotatedRect rect1, rect2;
		if (rect.size.width > rect.size.height)
		{
			Point2f newSize = rectPoints[3] - rectPoints[0];
			newSize.x *= 0.4;
			newSize.y *= 0.4;
			Point2f newPoint = rectPoints[0] + newSize;
			std::vector<Point> newPoints;
			newPoints.push_back(newPoint);
			newPoints.push_back(rectPoints[0]);
			newPoints.push_back(rectPoints[1]);
			rect1 = minAreaRect(newPoints);
			
			newPoint = rectPoints[3] - newSize;
			newPoints.clear();
			newPoints.push_back(newPoint);
			newPoints.push_back(rectPoints[2]);
			newPoints.push_back(rectPoints[3]);
			rect2 = minAreaRect(newPoints);
		}
		else
		{
			Point2f newSize = rectPoints[1] - rectPoints[0];
			newSize.x *= 0.4;
			newSize.y *= 0.4;
			Point2f newPoint = rectPoints[0] + newSize;
			std::vector<Point> newPoints;
			newPoints.push_back(newPoint);
			newPoints.push_back(rectPoints[0]);
			newPoints.push_back(rectPoints[3]);
			rect2 = minAreaRect(newPoints);
			
			newPoint = rectPoints[1] - newSize;
			newPoints.clear();
			newPoints.push_back(newPoint);
			newPoints.push_back(rectPoints[1]);
			newPoints.push_back(rectPoints[2]);
			rect1 = minAreaRect(newPoints);
		}
		Mat binaryMat;
		cvtColor(cropped, binaryMat, COLOR_BGR2GRAY);
		threshold(binaryMat, binaryMat, 100, 255, THRESH_BINARY_INV);
		/*imshow("binary", binaryMat);
		waitKey();*/
		vector<Point> nonZeroPoints;
		findNonZero(binaryMat, nonZeroPoints);
		Rect nonZeroRect = boundingRect(nonZeroPoints);
		Mat cropped1 = cropped(Rect(0,0,cropped.cols, nonZeroRect.y + nonZeroRect.height*0.4));
		Mat cropped2 = cropped(Rect(0, nonZeroRect.y + nonZeroRect.height*0.6, cropped.cols, cropped.rows - (nonZeroRect.y + nonZeroRect.height*0.6)));
		ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
		ocr->SetImage(cropped1.data, cropped1.cols, cropped1.rows, 3, cropped1.step);
		string str1 = ocr->GetUTF8Text();
		replaceChar(str1);
		ocr->SetImage(cropped2.data, cropped2.cols, cropped2.rows, 3, cropped2.step);
		string str2 = ocr->GetUTF8Text();
		replaceChar(str2);
		/*imshow("cropped1", cropped1);
		imshow("cropped2", cropped2);
		waitKey();*/
		if (!str1.empty())
		{
			putText(outTextMat, str1, rect1.center, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
			outText.push_back(pair<string, RotatedRect>(str1, rect1));
		}
		if (!str2.empty())
		{
			putText(outTextMat, str2, rect2.center, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
			outText.push_back(pair<string, RotatedRect>(str2, rect2));
		}
		//waitKey(0);
	}

  //  imshow("FFFFFFFFFFFFF", outTextMatShow);
//#ifdef SHOW_DEBUG
	string newPath = path + "TextMat.jpg";
	imwrite(newPath, outTextMat);
//#endif 
}

/**********************************************************************
 *  main()
 *
 **********************************************************************/
//
bool detectText(Mat &src, vector<pair<string, RotatedRect>> &outText, std::string path) {
	// Create Tesseract object
	tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
	// Initialize tesseract to use English (eng) and the LSTM OCR engine. 
	ocr->Init(g_traning_data_path.c_str(), "ngi+eng", tesseract::OEM_DEFAULT);
	// Set Page segmentation mode to PSM_AUTO (3)
	ocr->SetPageSegMode(tesseract::PSM_AUTO);
	/*ocr->SetVariable("edges_max_children_per_outline", "0");
	ocr->SetVariable("edges_max_children_layers", "0");
	ocr->SetVariable("edges_children_per_grandchild", "1");
	ocr->SetVariable("edges_children_count_limit", "2");*/
	int i;
	ocr->GetIntVariable("edges_max_children_per_outline", &i);
	//ocr->SetVariable("tessedit_char_whitelist", "φ0123456789abcdefjhijklmnopqrstuvwxyzABCDEFJHIJKLMNOPQRSTUVWXYZ.,+-");
	//string outTextStr;
	//ocr->SetImage(src.data, src.cols, src.rows, 3, src.step);

	// Run Tesseract OCR on image
	//outTextStr = string(ocr->GetUTF8Text());
	//std::wstring sLogLevel = utf8_to_utf16(outTextStr);
	/*std::wstring_convert<std::codecvt_utf8_utf16<char16_t>> converter;
	std::wstring wstr = wstring(converter.from_bytes(outTextStr));*/
	double scale = 640.0 / src.size().width;
	//resize(src, src, cv::Size(), scale, scale);
	/*namedWindow("Original image", CV_WINDOW_AUTOSIZE);
	imshow("Original image", src);*/
	vector<RotatedRect> filteredRects;
	findCharacterRects(src, filteredRects, path);

   /* Mat rectMat1 = src.clone();
    for (RotatedRect &rrec : filteredRects) {
        drawRotatedRectangle(rectMat1, rrec, Scalar(0, 255, 255));
    }
    namedWindow("detected_contour", CV_WINDOW_NORMAL);
    setWindowProperty("detected_contour", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    imshow("detected_contour", rectMat1);*/
   // waitKey(0);
	//group contours
	vector<vector<RotatedRect>> groupedRects;
	groupCharacterRects(filteredRects, groupedRects, src);

    /*Mat rectMat2 = src.clone();
    for (vector<RotatedRect> &rrecg : groupedRects) {
        int r = rand() % 255;
        int g = rand() % 255;
        int b = rand() % 255;
        for (RotatedRect &rrec : rrecg) {
            drawRotatedRectangle(rectMat2, rrec, Scalar(b, g, r));
        }
    }

    imshow("detected_contour2", rectMat2);*/
    //waitKey(0);

	findTextOfLine(src, groupedRects, ocr, outText, path);
	delete(ocr);
	return true;
}
