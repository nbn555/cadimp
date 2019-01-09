#include "shape_detection.h"
#include <math.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;
//#include <opencv2/ximgproc.hpp>
void thinningIteration(cv::Mat &img, int iter)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *nw, *no, *ne; // north (pAbove)
	uchar *we, *me, *ea;
	uchar *sw, *so, *se; // south (pBelow)

	uchar *pDst;

	// initialize row pointers
	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y)
	{
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x)
		{
			// shift col pointers left by one (scan left to right)
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);

			int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
				(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
				(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
				(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
			int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
			int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				pDst[x] = 1;
		}
	}

	img &= ~marker;
}

void thinning(const cv::Mat &src, cv::Mat &dst)
{
	dst = src.clone();
	dst /= 255; // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do
	{
		thinningIteration(dst, 0);
		thinningIteration(dst, 1);
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	dst *= 255;
}

void getIntersections(Mat src, std::vector<cv::Point> &vtIntersectionPoints, int flag)
{
	cv::Mat gray_image(src.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat thinning_image(src.size(), CV_8UC1, Scalar(0));
	std::vector<std::vector<Point>> vtContours;
	std::vector<Vec4i> vtHierarchy;

	// flag is
	// 0 - use thinning method
	if (flag == 0)
	{
		cv::Mat density(src.size(), CV_8UC1, cv::Scalar(0));
		cv::Mat kernel(3, 3, CV_8UC1, Scalar(1));
		//auto nRows = src.rows;

		if (src.channels() > 1)
		{
			cv::cvtColor(src, gray_image, COLOR_BGR2GRAY);
			gray_image = cv::Scalar::all(255) - gray_image;
			cv::threshold(gray_image, gray_image, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		}
		else
		{
			gray_image = src;
		}

		thinning(gray_image, thinning_image);
		//cv::ximgproc::thinning(gray_image, thinning_image, cv::ximgproc::THINNING_GUOHALL);
		cv::filter2D(thinning_image / 255, density, -1, kernel);
		cv::threshold(density, density, 3, 255, CV_THRESH_BINARY);
		cv::findContours(density, vtContours, vtHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	}
	// 1 - use corner Harris finding method. It's for getting center of circle. Some intersections is missing.
	else if (flag == 1)
	{
		auto nBlockSize = 5;
		cv::Mat dst, dst_norm;
		if (src.channels() > 1)
		{
			cv::cvtColor(src, gray_image, COLOR_BGR2GRAY);
			gray_image = cv::Scalar::all(255) - gray_image;
		}
		else
		{
			gray_image = src;
		}

		thinning(gray_image, thinning_image);
		// To increase accuracy, must use the following function
		//cv::ximgproc::thinning(gray_image, thinning_image, cv::ximgproc::THINNING_GUOHALL);
		cv::cornerHarris(thinning_image, dst, nBlockSize, 3, 0.04);
		cv::normalize(dst, dst_norm, 0, 255, cv::NormTypes::NORM_MINMAX, CV_8UC1);

		cv::threshold(dst_norm, dst_norm, 120, 255, CV_THRESH_BINARY);
		cv::findContours(dst_norm, vtContours, vtHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	}

	for (auto &&cnt : vtContours)
	{
		if (cnt.size() == 1)
		{
			vtIntersectionPoints.push_back(cnt[0]);
		}
		else
		{
			auto moment = cv::moments(cnt);
			if (moment.m00 != 0.0)
			{
				cv::Point point;
				point.x = int(moment.m10 / moment.m00);
				point.y = int(moment.m01 / moment.m00);
				//point.y = int(nRows - (moment.m01 / moment.m00));
				vtIntersectionPoints.push_back(point);
			}
		}
	}
}

#if 1
// Calculate distance between 2 points
double getDistance(const Point2f &point1, const Point2f &point2)
{
	return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
}
double getDistance2(const Point2f &point1, const Point2f &point2)
{
	return ((point1.x - point2.x)*(point1.x - point2.x) + (point1.y - point2.y)*(point1.y - point2.y));
}
// Calculate middle point
cv::Point2f getMidPoint(const cv::Point2f &p1, const cv::Point2f &p2)
{
	cv::Point2f mid_point;
	mid_point.x = (p1.x + p2.x)*0.5;
	mid_point.y = (p1.y + p2.y)*0.5;
	return mid_point;
}

#if 0
void detectLines(Mat src, vector<Vec4i> &lines)
{
	cv::Mat gray_image(src.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat edges_image(src.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat connected_image(src.size(), CV_8UC1, cv::Scalar(0));
	int nRows = src.rows;
	//int nNumberOfIteration = 3;

	cv::cvtColor(src, gray_image, CV_BGR2GRAY);
	gray_image = cv::Scalar::all(255) - gray_image;
	cv::threshold(gray_image, gray_image, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	//cv::GaussianBlur(gray_image, gray_image, cv::Size(3, 3), 0);

	auto nLabels = cv::connectedComponents(gray_image, connected_image);
	for (auto i = 1; i < nLabels; i++)
	{
		cv::Mat dst_image(src.size(), CV_8UC1, cv::Scalar(0));
		cv::inRange(connected_image, cv::Scalar(i), cv::Scalar(i), dst_image);

		//Step 1: Get all separate lines
		std::vector<std::vector<Point>> vtContours;
		std::vector<Vec4i> vtHierarchy;
		cv::Mat temp_image(dst_image.size(), dst_image.type(), cv::Scalar(0));
		bool check = false;

		cv::findContours(dst_image, vtContours, vtHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
		for (auto &&cnt : vtContours)
		{
			if (cv::arcLength(cnt, false) >= 5)
			{
				auto rect = cv::minAreaRect(cv::Mat(cnt));
				auto index = 0;
				cv::Point2f rect_points[4];
				rect.points(rect_points);

				if (getDistance(rect_points[0], rect_points[1]) >= getDistance(rect_points[1], rect_points[2]))
				{
					index = 1;
				}
				auto dDistance = getDistance(rect_points[index], rect_points[index + 1]);
				if (dDistance < 7.0 && dDistance >= 1.0)
				{
					if (getDistance(rect_points[index], rect_points[index + 1]) < 10.0)
					{
						std::vector<cv::Point> vtRectCnt;
						for (auto &&point : rect_points)
						{
							vtRectCnt.push_back(cv::Point((int)point.x, (int)point.y));
						}
						vtRectCnt.push_back(cv::Point((int)rect_points[0].x, (int)rect_points[0].y));
						std::vector<std::vector<Point>> vtCnt;
						vtCnt.push_back(vtRectCnt);
						cv::drawContours(temp_image, vtCnt, 0, cv::Scalar(255), -1);

						check = true;
						auto p1 = getMidPoint(rect_points[index], rect_points[index + 1]);
						auto p2 = getMidPoint(rect_points[index + 2], rect_points[(index + 3) % 4]);
#if 0
						sline l;
						l.p1.x = p1.x;
						l.p1.y = nRows - p1.y;
						l.p2.x = p2.x;
						l.p2.y = nRows - p2.y;
						lines.push_back(l);
#else
						lines.push_back(Vec4i(p1.x, p1.y, p2.x, p2.y));
#endif
					}
				}
			}
		}

		if (check)
		{
			cv::bitwise_and(dst_image, temp_image, temp_image);
			dst_image = dst_image - temp_image;
		}

		// Step 2:
		std::vector<cv::Vec4i> vtlines;
		//cv::Mat gray_temp_image(gray_image.size(), gray_image.type(), Scalar(0));

		//		cv::ximgproc::thinning(gray_image, temp_image);
		//thinning(gray_image, temp_image);
		cv::HoughLinesP(dst_image, vtlines, 1, CV_PI / 180, 5, 5, 5);
		if (vtlines.size() > 0)
		{
#if 0
			for (auto &&line : vtlines)
			{
				sline l;
				l.p1.x = line[0];
				l.p1.y = nRows - line[1];
				l.p2.x = line[2];
				l.p2.y = nRows - line[3];
				lines.push_back(l);
				cv::line(gray_temp_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 3);
			}
#else
			lines.insert(lines.end(), vtlines.begin(), vtlines.end());
			/*for (auto &&line : vtlines)
			{
			cv::line(gray_temp_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 3);
			}*/
#endif
		}
	}

	//	for (auto i = 0; i < nNumberOfIteration; i++)
	//	{
	//		std::vector<std::vector<Point>> vtContours;
	//		std::vector<Vec4i> vtHierarchy;
	//		cv::Mat temp_image(gray_image.size(), gray_image.type(), cv::Scalar(0));
	//
	//		// Step 1: Get all separate lines
	//		cv::Canny(gray_image, edges_image, 50, 150);
	//		cv::findContours(edges_image, vtContours, vtHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	//		auto element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//		for (auto &&cnt : vtContours)
	//		{
	//			auto rect = cv::minAreaRect(cv::Mat(cnt));
	//			auto index = 0;
	//			cv::Point2f rect_points[4];
	//			rect.points(rect_points);
	//			if (getDistance(rect_points[0], rect_points[1]) >= getDistance(rect_points[1], rect_points[2]))
	//			{
	//				index = 1;
	//			}
	//
	//			if (getDistance(rect_points[index], rect_points[index + 1]) < 10.0)
	//			{
	//				std::vector<cv::Point> vtRectCnt;
	//				for (auto &&point : rect_points)
	//				{
	//					vtRectCnt.push_back(cv::Point((int)point.x, (int)point.y));
	//				}
	//				vtRectCnt.push_back(cv::Point((int)rect_points[0].x, (int)rect_points[0].y));
	//				std::vector<std::vector<Point>> vtCnt;
	//				vtCnt.push_back(vtRectCnt);
	//				cv::drawContours(temp_image, vtCnt, 0, cv::Scalar(255), -1);
	//
	//				auto p1 = getMidPoint(rect_points[index], rect_points[index + 1]);
	//				auto p2 = getMidPoint(rect_points[index + 2], rect_points[(index + 3) % 4]);
	//#if 0
	//				sline l;
	//				l.p1.x = p1.x;
	//				l.p1.y = nRows - p1.y;
	//				l.p2.x = p2.x;
	//				l.p2.y = nRows - p2.y;
	//				lines.push_back(l);
	//#else
	//                lines.push_back(Vec4i(p1.x,p1.y,p2.x,p2.y));
	//#endif
	//			}
	//		}
	//		// Remove detected lines
	//		cv::dilate(temp_image, temp_image, element);
	//		cv::bitwise_and(gray_image, temp_image, temp_image);
	//		gray_image = gray_image - temp_image;
	//
	//		// Step 2:
	//		std::vector<cv::Vec4i> vtlines;
	//		cv::Mat gray_temp_image(gray_image.size(), gray_image.type(), Scalar(0));
	//
	////		cv::ximgproc::thinning(gray_image, temp_image);
	//        thinning(gray_image, temp_image);
	//		cv::HoughLinesP(temp_image, vtlines, 1, CV_PI / 180, 30, 20, 10);
	//		if (vtlines.size() > 0)
	//		{
	//#if 0
	//			for (auto &&line : vtlines)
	//			{
	//				sline l;
	//				l.p1.x = line[0];
	//				l.p1.y = nRows - line[1];
	//				l.p2.x = line[2];
	//				l.p2.y = nRows - line[3];
	//				lines.push_back(l);
	//				cv::line(gray_temp_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 3);
	//			}
	//#else
	//            lines.insert(lines.end(), vtlines.begin(), vtlines.end());
	//            /*for (auto &&line : vtlines)
	//            {
	//                cv::line(gray_temp_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 3);
	//            }*/
	//#endif
	//			// Remove detected lines
	//			cv::dilate(gray_temp_image, gray_temp_image, element);
	//			cv::bitwise_and(gray_image, gray_temp_image, temp_image);
	//			gray_image = gray_image - temp_image;
	//		}
	//	}
}
#else

void detectLines(Mat src, vector<Vec4i> &lines)
{
	cv::Mat gray_image(src.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat edges_image(src.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat connected_image(src.size(), CV_8UC1, cv::Scalar(0));
	int nRows = src.rows;
	//int nNumberOfIteration = 3;

	cv::cvtColor(src, gray_image, CV_BGR2GRAY);
	gray_image = cv::Scalar::all(255) - gray_image;
	cv::threshold(gray_image, gray_image, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	//cv::GaussianBlur(gray_image, gray_image, cv::Size(3, 3), 0);

	auto nLabels = cv::connectedComponents(gray_image, connected_image);

	cv::Mat dst_image(src.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp_image(dst_image.size(), dst_image.type(), cv::Scalar(0));
	std::vector<std::vector<Point>> vtContours;
	std::vector<Vec4i> vtHierarchy;
	cv::Point2f rect_points[4];
	std::vector<std::vector<Point>> vtCnt(1);
	vtCnt[0].resize(5);
	std::vector<cv::Point>* vtRectCnt = &vtCnt[0];
	double distance2;
	for (auto i = 1; i < nLabels; i++)
	{

		cv::inRange(connected_image, cv::Scalar(i), cv::Scalar(i), dst_image);
		//Step 1: Get all separate lines

		bool check = false;
		vtContours.clear();
		vtHierarchy.clear();
		cv::findContours(dst_image, vtContours, vtHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
		for (auto &&cnt : vtContours)
		{
			if (cv::arcLength(cnt, false) >= 5)
			{
				auto rect = cv::minAreaRect(cv::Mat(cnt));
				auto index = 0;

				rect.points(rect_points);

				if (getDistance2(rect_points[0], rect_points[1]) >= getDistance2(rect_points[1], rect_points[2]))
				{
					index = 1;
				}
				auto dDistance = getDistance2(rect_points[index], rect_points[index + 1]);
				if (dDistance < 49.0 && dDistance >= 1.0)
				{
					/*if (getDistance(rect_points[index], rect_points[index + 1]) < 10.0)
					{*/

					for (int i1 = 0; i1 < 4; ++i1)
					{
						vtRectCnt->at(i1) = cv::Point((int)rect_points[i1].x, (int)rect_points[i1].y);
					}
					vtRectCnt->at(4) = (cv::Point((int)rect_points[0].x, (int)rect_points[0].y));

					cv::drawContours(temp_image, vtCnt, 0, cv::Scalar(255), -1);
					check = true;
					auto p1 = getMidPoint(rect_points[index], rect_points[index + 1]);
					auto p2 = getMidPoint(rect_points[index + 2], rect_points[(index + 3) % 4]);

					lines.push_back(Vec4i(p1.x, p1.y, p2.x, p2.y));
					//  }
				}
			}
		}

		if (check)
		{
			cv::bitwise_and(dst_image, temp_image, temp_image);
			dst_image = dst_image - temp_image;
		}

		// Step 2:
		std::vector<cv::Vec4i> vtlines;
		//cv::Mat gray_temp_image(gray_image.size(), gray_image.type(), Scalar(0));

		//		cv::ximgproc::thinning(gray_image, temp_image);
		//thinning(gray_image, temp_image);
		cv::HoughLinesP(dst_image, vtlines, 1, CV_PI / 180, 5, 5, 5);
		if (vtlines.size() > 0)
		{
			lines.insert(lines.end(), vtlines.begin(), vtlines.end());
		}
	}

}

void detectArrows(const Mat &src, vector<vector<Point2f>> &vtArrows, const double &standard_area, const double &standard_ratio)
{
	printf("%f\n", standard_area);
	Mat gray_image(src.size(), CV_8UC1, Scalar(0));
	Mat black_hat_image(src.size(), CV_8UC1, Scalar(0));
	Mat kernel1(2, 2, CV_8UC1, Scalar(1));
	Mat kernel2(3, 3, CV_8UC1, Scalar(1));
	vector<vector<Point>> vtContours;
	vector<Vec4i> vtHierarchy;

	if (src.channels() > 2)
	{
		cvtColor(src, gray_image, COLOR_BGR2GRAY);
	}
	else
	{
		gray_image = src;
	}

	threshold(gray_image, gray_image, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	// Apply the following formula: inverted threshold image - black hat of image.
	dilate(gray_image, black_hat_image, kernel1);
	erode(black_hat_image, black_hat_image, kernel1);
	black_hat_image = (255 - gray_image) - (black_hat_image - gray_image);

	erode(black_hat_image, black_hat_image, kernel2);
	dilate(black_hat_image, black_hat_image, kernel2);

	cv::findContours(black_hat_image, vtContours, vtHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	auto index = 0;
	for (auto &&cnt : vtContours)
	{
		vector<Point2f> triangle;
		minEnclosingTriangle(cnt, triangle);
		if (triangle.size() > 0)
		{
			auto cnt_area = contourArea(cnt);
			auto triangle_area = abs(triangle[0].x * (triangle[1].y - triangle[2].y) +
				triangle[1].x * (triangle[2].y - triangle[0].y) +
				triangle[2].x * (triangle[0].y - triangle[1].y)) /
				2;
			if (triangle_area > 0.0)
			{
				// Get ratio between boundary triangle area and region area
				auto ratio = cnt_area / triangle_area;
				// Compare with conditions.
				if (ratio >= standard_ratio && cnt_area <= standard_area)
				{
					RotatedRect rect = minAreaRect(cnt);
					cv::Point2f vertices2f[4];
					rect.points(vertices2f);
					auto edge_1 = getDistance2(vertices2f[0], vertices2f[1]);
					auto edge_2 = getDistance2(vertices2f[2], vertices2f[1]);
					auto max_value = std::max(edge_1, edge_2);
					auto min_value = std::min(edge_1, edge_2);
					if (max_value / min_value <= 16.0)
					{
						vtArrows.push_back(triangle);
					}
				}
			}
		}
	}
}

#endif
#else

void getLines(Mat src, std::vector<sline> &lines)
{
	cv::Mat invSrc = cv::Scalar::all(255) - src;
	cv::Mat bw;
	cv::cvtColor(invSrc, bw, CV_BGR2GRAY);
	//GaussianBlur(bw, bw, Size(3, 3), 0, 0 );
	cv::threshold(bw, bw, 30, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	Mat result;
	cv::adaptiveThreshold(bw, result, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 23, 0);

	/*imshow("result", result);

	cv::imshow("src", invSrc);
	cv::imshow("dst", bw);
	cv::waitKey();*/

	Mat cdst;
	cvtColor(bw, cdst, COLOR_GRAY2BGR);
	Mat cdstP = cdst.clone();

	//Standard Hough Line Transform
	//std::vector<Vec2f> liness; // will hold the results of the detection
	//HoughLines(bw, liness, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
	//												  // Draw the lines
	//for (size_t i = 0; i < liness.size(); i++)
	//{
	//	float rho = liness[i][0], theta = liness[i][1];
	//	Point pt1, pt2;
	//	double a = cos(theta), b = sin(theta);
	//	double x0 = a*rho, y0 = b*rho;
	//	pt1.x = cvRound(x0 + 1000 * (-b));
	//	pt1.y = cvRound(y0 + 1000 * (a));
	//	pt2.x = cvRound(x0 - 1000 * (-b));
	//	pt2.y = cvRound(y0 - 1000 * (a));
	//	line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	//}
	int ih = src.rows;
	double lw = 1.0;
	if (ih > 1000)
		lw = 2.0;

	// Probabilistic Line Transform
	vector<Vec4i> linesP;								 // will hold the results of the detection
	HoughLinesP(bw, linesP, lw, CV_PI / 180, 50, 30, 2); // runs the actual detection
														 // Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
		sline lin;

		lin.p1 = Point2dd(l[0], ih - l[1]);
		lin.p2 = Point2dd(l[2], ih - l[3]);
		lines.push_back(lin);
	}
}
#endif