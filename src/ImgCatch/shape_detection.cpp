#include "shape_detection.h"


void thinningIteration(cv::Mat& img, int iter)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *nw, *no, *ne;    // north (pAbove)
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;    // south (pBelow)

	uchar *pDst;

	// initialize row pointers
	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y) {
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

		for (x = 1; x < img.cols - 1; ++x) {
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


void thinning(const cv::Mat& src, cv::Mat& dst)
{
	dst = src.clone();
	dst /= 255;         // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningIteration(dst, 0);
		thinningIteration(dst, 1);
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	dst *= 255;
}

#if 1

void getLines(Mat src, std::vector<sline>& lines)
{
	cv::Mat invSrc = cv::Scalar::all(255) - src;
	cv::Mat bw;
	cv::cvtColor(invSrc, bw, CV_BGR2GRAY);
	//GaussianBlur(bw, bw, Size(3, 3), 0, 0 );
    double threshold = 80.0;
	cv::threshold(bw, bw, threshold, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
   // cv::imshow("bw", bw);
	//Mat result;
	//cv::adaptiveThreshold(bw, bw, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 21, 0);
   // imshow("result", result);
	/*imshow("result", result);

	cv::imshow("src", invSrc);
	cv::imshow("dst", bw);
	cv::waitKey();*/


	//Mat cdst;
	//cvtColor(bw, cdst, COLOR_GRAY2BGR);
	//Mat cdstP = cdst.clone();


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
    int sizemax =max(ih, src.cols);
    double minLineLength = 5;
    double maxLineGap = 0;
    double lw = 1.0;
    if (sizemax > 3500) {
        lw = 3.0;
        minLineLength = 20;
        maxLineGap = 2;
    }
    else if (sizemax > 2000) {
        lw = 2.0;
        minLineLength = 10;
        maxLineGap = 1;
    }

   // cv::imshow("bbbbw", bw);

	// Probabilistic Line Transform
	vector<Vec4i> linesP; // will hold the results of the detection
	HoughLinesP(bw, linesP, lw, CV_PI / 180, threshold, minLineLength, maxLineGap); // runs the actual detection
														// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		//line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
		sline lin;

		lin.p1 = Point2Dd(l[0], ih-l[1]);
		lin.p2 = Point2Dd(l[2], ih-l[3]);
		lines.push_back(lin);
	}
	// Show results
	/*imshow("Source", src);
	imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
	imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
	imwrite("line1.png", cdstP);
	imwrite("line0.png", cdst);
	waitKey(0);*/
}

#else 


void getLines(Mat src, std::vector<sline>& lines)
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
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(bw, linesP, lw, CV_PI / 180, 50, 30, 2); // runs the actual detection
                                                         // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
        sline lin;

        lin.p1 = Point2Dd(l[0], ih - l[1]);
        lin.p2 = Point2Dd(l[2], ih - l[3]);
        lines.push_back(lin);
    }
}
#endif