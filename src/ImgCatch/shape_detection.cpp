#include "shape_detection.h"
#include <math.h>

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

#if 1
// Calculate distance between 2 points
double getDistance(const Point2f &point1, const Point2f &point2)
{
	return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
}

// Calculate middle point
cv::Point2f getMidPoint(const cv::Point2f &p1, const cv::Point2f &p2)
{
	cv::Point2f mid_point;
	mid_point.x = (p1.x + p2.x) / 2;
	mid_point.y = (p1.y + p2.y) / 2;
	return mid_point;
}

void getLines(Mat src, std::vector<sline> &lines)
{
	cv::Mat gray_image(src.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat edges_image(src.size(), CV_8UC1, cv::Scalar(0));
	int nRows = src.rows;
	int nNumberOfIteration = 3;

	cv::cvtColor(src, gray_image, CV_BGR2GRAY);
	gray_image = cv::Scalar::all(255) - gray_image;
	cv::threshold(gray_image, gray_image, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	cv::GaussianBlur(gray_image, gray_image, cv::Size(3, 3), 0);

	for (auto i = 0; i < nNumberOfIteration; i++)
	{
		std::vector<std::vector<Point>> vtContours;
		std::vector<Vec4i> vtHierarchy;
		cv::Mat temp_image(gray_image.size(), gray_image.type(), cv::Scalar(0));

		// Step 1: Get all separate lines
		cv::Canny(gray_image, edges_image, 50, 150);
		cv::findContours(edges_image, vtContours, vtHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
		auto element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		for (auto &&cnt : vtContours)
		{
			auto rect = cv::minAreaRect(cv::Mat(cnt));
			auto index = 0;
			cv::Point2f rect_points[4];
			rect.points(rect_points);
			if (getDistance(rect_points[0], rect_points[1]) >= getDistance(rect_points[1], rect_points[2]))
			{
				index = 1;
			}

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

				sline l;
				auto p1 = getMidPoint(rect_points[index], rect_points[index + 1]);
				auto p2 = getMidPoint(rect_points[index + 2], rect_points[(index + 3) % 4]);
				l.p1.x = p1.x;
				l.p1.y = nRows - p1.y;
				l.p2.x = p2.x;
				l.p2.y = nRows - p2.y;
				lines.push_back(l);
			}
		}
		// Remove detected lines
		cv::dilate(temp_image, temp_image, element);
		cv::bitwise_and(gray_image, temp_image, temp_image);
		gray_image = gray_image - temp_image;

		// Step 2:
		std::vector<cv::Vec4i> vtlines;
		cv::Mat gray_temp_image(gray_image.size(), gray_image.type(), Scalar(0));

		cv::ximgproc::thinning(gray_image, temp_image);
		cv::HoughLinesP(temp_image, vtlines, 1, CV_PI / 180, 30, 20, 10);
		if (vtlines.size() > 0)
		{
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
			// Remove detected lines
			cv::dilate(gray_temp_image, gray_temp_image, element);
			cv::bitwise_and(gray_image, gray_temp_image, temp_image);
			gray_image = gray_image - temp_image;
		}
	}
}

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