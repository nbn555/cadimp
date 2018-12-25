#include"LiProcess.h"

#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "dirent.h"
#include "shape_detection.h"

void removeBorder(Mat& src, Mat &removedBorderMat) {
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);
	Mat blurMat;
	GaussianBlur(gray, blurMat, Size(5, 5), 0);
	Mat binaryMat;
	threshold(gray, binaryMat, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
	vector<vector<Point2i>> contours;
	vector<Vec4i> hierarchy;
	findContours(binaryMat, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	removedBorderMat = src.clone();
	for (size_t i = 0; i < contours.size(); i++)
	{
		
		if (hierarchy[i][3] == -1) {
			continue;
		}
		Rect rect = minAreaRect(contours[i]).boundingRect();
		if (rect.height < 0.8 * src.rows || rect.width < 0.8 * src.cols)
		{
			continue;
		}

		//Erode rectangle
		rect.x += 10;
		rect.y += 10;
		rect.width -= 20;
		rect.height -= 20;
		if (rect.width <= 0 || rect.x + rect.width >= src.cols || rect.y <= 0 || rect.y + rect.height >= src.rows) {
			continue;
		}
		removedBorderMat.setTo(Scalar::all(255));
		src(rect).copyTo(removedBorderMat(rect));
		
		//imwrite("removedBorderMat.jpg", removedBorderMat);
		//Mat contourMat = src.clone();
		//drawContours(contourMat, contours, i, Scalar(0, 0, 255), 5);
		//namedWindow("contourMat", CV_WINDOW_NORMAL);
		//setWindowProperty("contourMat", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		//imshow("contourMat", contourMat/*removedBorderMat*/);
		//waitKey();
		return;
		//imwrite("removedMat.jpg", contourMat);
	}
}
void testFolder(const string &path) {
	DIR *pDIR;
	struct dirent *entry;
	//Mat original_mat;
	string full_path;

	if (pDIR = opendir(path.c_str())) {
		while (entry = readdir(pDIR)) {
			if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
				string newPath = path + "\\" + entry->d_name;
				if (entry->d_type == DT_DIR) {
					testFolder(newPath);
					continue;
				}
				Mat src, src_gray;

				/// Read the image
				src = imread(newPath);
				if (!src.data)
				{
					continue;
				}
				vector<Vec3f> outCircles;
				//detectCircle(src, outCircles, newPath);
			}
		}
		closedir(pDIR);
		delete entry;
	}
}

void detectCircle1(Mat &src, vector<Point> intersectionPoints, vector<Vec3f> &outCircles) {
	if (!src.data)
	{
		return;
	}

	if (src.cols <= 20 || src.rows < 20)
	{
		return;
	}

	int param2 = src.cols > 1000 ? 100 : (src.cols > 200 ? 50 : 30);

	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	outCircles.clear();
	vector<Vec3f> circles;
	int iterator = 0;
	do
	{
		std::vector<Point> newIntersectionPoints = intersectionPoints;
		HoughCircles(src_gray, circles, newIntersectionPoints, HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, param2, 0, 0);

		/// Draw the circles detected
		for (size_t i = 0; i < circles.size(); i++)
		{
			outCircles.push_back(circles[i]);
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(src_gray, center, radius, Scalar::all(0), 15, 8, 0);
		}
		iterator++;
	} while (!circles.empty() && iterator < 10);
	//for (size_t i = 0; i < outCircles.size(); i++)
	//{
	//	/*outCircles[i][0] = outCircles[i][0] / scale;
	//	outCircles[i][1] = outCircles[i][1] / scale;
	//	outCircles[i][2] = outCircles[i][2] / scale;*/
	//	Point center(cvRound(outCircles[i][0]), cvRound(outCircles[i][1]));
	//	int radius = cvRound(outCircles[i][2]);
	//	// circle center
	//	circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
	//	// circle outline
	//	circle(src, center, radius, Scalar(0, 0, 255), 9, 8, 0);
	//}
	//imshow("src", src);
	//waitKey(0);
	/*namedWindow("circles", CV_WINDOW_NORMAL);
	setWindowProperty("circles", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);*/
	//path += "circle.jpg";
	//imwrite(path, src);
}

void detectCircle2(Mat &src, vector<Vec3f> &outCircles, vector<Point> intersectionPoints, string path) {
	if (!src.data)
	{
		return;
	}
	Mat resizedMat;
	double scale = 1720.0 / src.size().width;
	//resizedMat = src.clone();
	resize(src, resizedMat, cv::Size(), scale, scale, cv::INTER_AREA);
	getIntersections(resizedMat, intersectionPoints);
	Mat intersectionPointMat = resizedMat.clone();
	for (size_t i = 0; i < intersectionPoints.size(); i++)
	{
		circle(intersectionPointMat, intersectionPoints[i], 4, Scalar(0, 0, 255), 2);
	}
	namedWindow("intersectionPointMat", CV_WINDOW_NORMAL);
	setWindowProperty("intersectionPointMat", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	imshow("intersectionPointMat", intersectionPointMat);
	waitKey();
	//namedWindow("Original image", CV_WINDOW_AUTOSIZE);
	//imshow("Original image", src);
	Mat src_gray;
	cvtColor(resizedMat, src_gray, CV_BGR2GRAY);
	Canny(src_gray, src_gray, 50, 150, 3);
	//namedWindow("src_gray", CV_WINDOW_NORMAL);
	//setWindowProperty("src_gray", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	//imshow("src_gray", src_gray);
	//waitKey(0);
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	outCircles.clear();
	int iterator = 0;
	vector<Vec3f> circles;
	do
	{
		circles.clear();
		/// Apply the Hough Transform to find the circles
		HoughCircles(src_gray, circles, intersectionPoints, HOUGH_GRADIENT, 1, src_gray.rows / 40, 200, 100, 0, 0);

		Mat intersectionPointMat = resizedMat.clone();
		for (size_t i = 0; i < intersectionPoints.size(); i++)
		{
			circle(intersectionPointMat, intersectionPoints[i], 4, Scalar(0, 0, 255), 2);
		}
		namedWindow("intersectionPointMat", CV_WINDOW_NORMAL);
		setWindowProperty("intersectionPointMat", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		imshow("intersectionPointMat", intersectionPointMat);
		waitKey();

		/// Draw the circles detected
		for (size_t i = 0; i < circles.size(); i++)
		{
			outCircles.push_back(circles[i]);
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(src_gray, center, radius, Scalar::all(0), 15, 8, 0);
		}
		iterator++;
	} while (!circles.empty() && iterator < 10);
	for (size_t i = 0; i < outCircles.size(); i++)
	{
		outCircles[i][0] = outCircles[i][0] / scale;
		outCircles[i][1] = outCircles[i][1] / scale;
		outCircles[i][2] = outCircles[i][2] / scale;
		Point center(cvRound(outCircles[i][0]), cvRound(outCircles[i][1]));
		int radius = cvRound(outCircles[i][2]);
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius, Scalar(0, 0, 255), 9, 8, 0);
	}

	/// Show your results
	namedWindow("circles", CV_WINDOW_NORMAL);
	setWindowProperty("circles", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	imshow("circles", src);
	waitKey(0);
	//namedWindow("circles", CV_WINDOW_NORMAL);
	//setWindowProperty("circles", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	//imshow("circles", src);
	/*path += "circle.jpg";
	imwrite(path, src);*/
	//waitKey(0);

}

void detectCircle(Mat &src, vector<Vec3f> &outCircles,vector<Point> intersectionPoints, string path) {
	if (!src.data)
	{
		return;
	}
	Mat resizedMat;
	double scale = 1720.0 / src.size().width;
	//resizedMat = src.clone();
	resize(src, resizedMat, cv::Size(), scale, scale, cv::INTER_AREA);
	getIntersections(resizedMat, intersectionPoints);
	Mat intersectionPointMat = resizedMat.clone();
	for (size_t i = 0; i < intersectionPoints.size(); i++)
	{
		circle(intersectionPointMat, intersectionPoints[i], 4, Scalar(0, 0, 255), 2);
	}
	namedWindow("intersectionPointMat", CV_WINDOW_NORMAL);
	setWindowProperty("intersectionPointMat", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	imshow("intersectionPointMat", intersectionPointMat);
	waitKey();
	//namedWindow("Original image", CV_WINDOW_AUTOSIZE);
	//imshow("Original image", src);
	Mat grayMat, cannyMat;
	cvtColor(resizedMat, grayMat, CV_BGR2GRAY);
	Canny(grayMat, cannyMat, 50, 150, 3);
	/*Mat dilateMat;
	Mat element = getStructuringElement(MORPH_RECT,Size(3, 3),Point(2,2));
	dilate(cannyMat, dilateMat, element);*/
	vector<vector<Point>> contours;
	findContours(cannyMat.clone(), contours, RETR_LIST, CHAIN_APPROX_NONE);
	Mat contourMat = Mat(cannyMat.size(), CV_8UC1, Scalar::all(0));
	drawContours(contourMat, contours, -1, Scalar::all(255));
	Mat blurMat;
	GaussianBlur(cannyMat, blurMat, Size(9, 9), 2, 2);
	//imshow("blurMat", blurMat);
	//imshow("contourMat", contourMat);
	//waitKey(0);
	Rect rect;
	Mat croppedMat;
	for (size_t contourId = 0; contourId < contours.size(); contourId++)
	{
		outCircles.clear();
		//drawContours(contourMat, contours, i, Scalar::all(255));

		rect = minAreaRect(contours[contourId]).boundingRect();
		rect.width += 40;
		rect.height += 40;
		rect.x -= 20;
		rect.y -= 20;
		if (rect.x <0)
		{
			rect.x = 0;
		}
		if (rect.x + rect.width > blurMat.cols - 1)
		{
			rect.width = blurMat.cols - rect.x - 1;
		}

		if (rect.y <0)
		{
			rect.y = 0;
		}
		if (rect.y + rect.height > blurMat.rows - 1)
		{
			rect.height = blurMat.rows - rect.y - 1;
		}
		if (rect.width < 50 || rect.height < 50)
		{
			continue;
		}
		/*imshow("contourMat", contourMat);
		waitKey(0);*/
		croppedMat = blurMat(rect).clone();
		/// Reduce the noise so we avoid false circle detection
		string croppedName = std::to_string(contourId) + ".jpg";
		cvtColor(croppedMat, croppedMat, CV_GRAY2BGR);
		std:vector<Point> newIntersectionPoints(intersectionPoints.size());
		for (size_t intersectionPointId = 0; intersectionPointId < intersectionPoints.size(); intersectionPointId++)
		{
			newIntersectionPoints[intersectionPointId].x = intersectionPoints[intersectionPointId].x - rect.x;
			newIntersectionPoints[intersectionPointId].y = intersectionPoints[intersectionPointId].y - rect.y;
		}
		imwrite(croppedName, croppedMat);
		vector<Vec3f> circles;
		detectCircle1(croppedMat, newIntersectionPoints, circles);
		int iterator = 0;
		for (size_t i = 0; i < circles.size(); i++)
		{

			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			//// circle center
			//circle(croppedMat, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			//// circle outline
			//circle(croppedMat, center, radius, Scalar(0, 0, 255), 9, 8, 0);
			Mat croppedCannyMat = cannyMat(rect).clone();
			Mat circleMat(croppedCannyMat.size(), CV_8UC1, Scalar::all(0));
			circle(circleMat, center, radius, Scalar::all(255));
			int circleLength = countNonZero(circleMat);
			Mat dilateMat;
			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(2, 2));
			dilate(croppedCannyMat, dilateMat, element);
			erode(dilateMat, dilateMat, element);
			//imshow("circleMat", circleMat);
			bitwise_and(circleMat, dilateMat, circleMat);
			/*imshow("dilateMat", dilateMat);
			imshow("bitwise_and", circleMat);
			waitKey(0);*/
			int innerCircleLength = countNonZero(circleMat);
			float innerCircleRatio = (float)innerCircleLength / circleLength;
			if (innerCircleRatio < 0.3)
			{
				continue;
			}
			circles[i][0] = (rect.x + circles[i][0]) / scale;
			circles[i][1] = (rect.y + circles[i][1]) / scale;
			circles[i][2] = circles[i][2] / scale;
			outCircles.push_back(circles[i]);
			center = Point(cvRound(circles[i][0]), cvRound(circles[i][1]));
			radius = cvRound(circles[i][2]);
			// circle center
			circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(src, center, radius, Scalar(0, 0, 255), 9, 8, 0);
		}

		/// Show your results
		/*namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
		imshow("Hough Circle Transform Demo", src);
		waitKey(0);*/
		/*namedWindow("circles", CV_WINDOW_NORMAL);
		setWindowProperty("circles", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);*/
		/*imshow("circles", croppedMat);
		string circleName = std::to_string(i) + "circle.jpg";
		imwrite(circleName, croppedMat);*/
		//waitKey(0);
	}
	path += "circles.jpg";
	imwrite(path, src);

}
// khoi tao ma tran 2 chieu kich thuoc size
void CreateIntMatrix(int **&matrix, CvSize size) {
	matrix = new int*[size.height];
	for (int iInd = 0; iInd < size.height; iInd++)
		matrix[iInd] = new int[size.width];
}
// giai phong ma tran 
void ReleaseIntMatrix(int **&matrix, int size) {
	for (int iInd = 0; iInd < size; iInd++)
		delete[] matrix[iInd];
}
/*****************************************************************************/
/**label_Neighbor: Tinh so lang gieng cung nhan cua (x,y)
 * @param label
 * [in]: Nhan diem anh
 * @param x,y
 * [in]: Toa do diem anh. 
 * @param d
 * [in]: gia tri nhan
 * @return: sum
 * so lang gieng cua (x,y) 
 *****************************************************************************/
int label_Neighbor(int **label, int y, int x, int d) {
	int sum = 0;	
	if (label[y][x-1] == d)	sum = sum + 1;
	if (label[y][x+1] == d)	sum = sum + 1;
	if (label[y-1][x] == d)	sum = sum + 1;
	if (label[y+1][x] == d)	sum = sum + 1;
	return sum; 
}
/*****************************************************************************/
/** Neighbor_3: kiem tra 1 point co la diem noi 3 thuc su 
 * @param label
 * [in]: Nhan diem anh
 * @param x,y
 * [in]: Toa do diem anh. 
 * @return: 
 * 1:  (x,y) la diem noi thuc su
 * 0:  xung quanh (x,y) la cac diem noi 3
 *****************************************************************************/
int Neighbor_3(int **label, int y, int x) {
	int sum = 0, t;	
	if (label[y][x-1] >= 0)	sum = sum + 1;
	if (label[y][x+1] >= 0)	sum = sum + 1;
	if (label[y-1][x] >= 0)	sum = sum + 1;
	if (label[y+1][x] >= 0)	sum = sum + 1;
	if (sum > 0)	t = 0;	
		else		t = 1;	
	return t;
}

int orient_Neighbor(int **label, int y, int x, int label_xy) {
	int sum, t1 = 0;
	if ((label[y-1][x] == label_xy) && (label[y+1][x] == label_xy))
		t1 = t1 + 1;
	if ((label[y][x-1] == label_xy) && (label[y][x+1] == label_xy))
		t1 = t1 + 1;
	if ((label[y][x-1] == label_xy) && (label[y+1][x] == label_xy))
		t1 = t1 + 1;
	if ((label[y][x+1] == label_xy) && (label[y+1][x] == label_xy))
		t1 = t1 + 1;
	if (t1 > 0)		sum = 0; 
		else		sum = 1;
	return sum;
}

/////////////////////////////////////
/*****************************************************************************/
/** intersection: xac dinh cac diem noi 
 * @param img
 * [in]: Anh.
 * @param CvPoint *point
 * [out]: Toa do cac diem anh.
 * @param t
 * [out]: So luong diem anh
 * @param label
 * [out]: Nhan cua diem anh 
 * @param tx, ty
 * [out]: Khoang cach hoanh do, tung do giua 2 diem noi 
 * @return: none  
 /** connect_point_3: Ket noi cac diem noi 3
 /** connect_point_34: Ket noi cac diem noi 3 voi noi 4 
 *****************************************************************************/
void intersection(IplImage *img,CvPoint *point, int &t, int **label, int **tx,int **ty) 
{
	int i, j, k, sum;
	CvSize Ssize;
	Ssize.width = img->width;
	Ssize.height = img->height;	
	int **data;    CreateIntMatrix(data ,Ssize);
	///  tap du lieu la diem anh ///
	for (i = 1; i< img->height-1; i++)
		for (j = 1; j< img->width-1; j++)
		{
			data[i][j] = 0;	label[i][j] = -2;		
			if (data(img,j,i) > 0)
			{
				point[t].x = j;		point[t].y = i;				
				data[i][j] = 255;	t = t+1;		
			}
		}	
	/// end ///
	CvPoint *point_3; point_3 = new CvPoint[t];
	CvPoint *point_34; point_34 = new CvPoint[t];
	int t_3 = 0, t_34 = 0;
	/// xac dinh giao diem 3,4///
	for (k = 1; k < t; k++)
	{
		j = point[k].x; i = point[k].y;			 
		label[i][j] = 0; tx[i][j] = 0; ty[i][j] = 0;
		sum = 0;	
		if (255 == data[i][j])
		{
			if (255 == data[i-1][j]) sum = sum+1;
			if (255 == data[i+1][j]) sum = sum+1;
			if (255 == data[i][j-1]) sum = sum+1;
			if (255 == data[i][j+1]) sum = sum+1;			 
		}		
		if (sum ==3)  
			{ 
				label[i][j] = -3;	  point_3[t_3] = point[k];	t_3 = t_3+1; 
				point_34[t_34] = point[k];	t_34 = t_34+1; 
			}        
		if (sum ==4)  	
			{ 
				label[i][j] = -4;
				point_34[t_34] = point[k];	t_34 = t_34+1; 
			}			       
	}		
	connect_point_3(point_3, t_3,label, tx, ty);	
	connect_point_34(point_34, t_34, label, tx, ty);	
	delete[] point_3;
	delete[] point_34;
	ReleaseIntMatrix(data,Ssize.height);
}
/////////////////////////////////////////
/*****************************************************************************/
/** connect_point_34 : ket noi diem noi 3 voi diem noi 4 o khoang cach gan 
 * @param CvPoint *point
 * [in]: Toa do cac diem anh.
 * @param t
 * [in]: So luong diem anh
 * @param label
 * [in_out]: Nhan cua diem anh 
 * @param tx, ty
 * [in_out]: Khoang cach hoanh do, tung do giua 2 diem noi 3 
 * @return: none  
 *****************************************************************************/
void connect_point_34(CvPoint *point, int t,int **label, int **tx, int **ty) 
{
	int i1, j1, i2, j2, i3, j3, i, j;
	for (i = 0; i < t; i++) 
	{
		j1 = point[i].x;		i1 = point[i].y;
		if ((label[i1][j1] == -3))
			for (j = 0; j < t; j++) 
			{
				j2 = point[j].x;		i2 = point[j].y;				
				if (label[i2][j2] == -4)			
					if ((abs(i1 - i2) <= pixel_connect_intersection)
							&& (abs(j1 - j2) <= pixel_connect_intersection)) 
					{
						int minx = min(j1,j2); int maxx = max (j1,j2);
						int miny = min(i1,i2); int maxy = max (i1,i2);
						for (i3 = miny; i3 <= maxy; i3++)
							for (j3 = minx; j3 <= maxx; j3++)								
								if (0 == label[i3][j3]) label[i3][j3] = -3;						
						tx[i1][j1] = abs(j1 - j2);	ty[i1][j1] = abs(i1 - i2);
						tx[i2][j2] = tx[i1][j1];	ty[i2][j2] = ty[i1][j1];
						break;
					}
			}
	}
}
/*****************************************************************************/
/** connect_point_3 : ket noi 2 diem noi 3 o khoang cach gan
 * @param CvPoint *point
 * [in]: Toa do cac diem anh.
 * @param t
 * [in]: So luong diem anh
 * @param label
 * [in_out]: Nhan cua diem anh 
 * @param tx, ty
 * [in_out]: Khoang cach hoanh do, tung do giua 2 diem noi 3 
 * @return: none  
 *****************************************************************************/
void connect_point_3(CvPoint *point, int t, int **label, int **tx, int **ty)
{
	int i1, j1, i2, j2, i, j;
	for (i = 0; i < t - 1; i++) 
	{
		j1 = point[i].x;	i1 = point[i].y;		
		for (j = i + 1; j < t; j++) 
		{
			j2 = point[j].x;		i2 = point[j].y;			
			if ((tx[i2][j2] == 0) && (ty[i2][j2] == 0))
				if ((abs(i1 - i2) <= pixel_connect_intersection)
						&& (abs(j1 - j2) <= pixel_connect_intersection)) 
				{		
					point3_point3(label,i1, j1, i2, j2);
					tx[i1][j1] = abs(j1 - j2);	ty[i1][j1] = abs(i1 - i2);
					tx[i2][j2] = tx[i1][j1];	ty[i2][j2] = ty[i1][j1];
					break;
				}
		}
		if ((label[i1][j1 + 1] == -3) && (label[i1 + 1][j1] == -3)&& (label[i1 + 1][j1 + 1] == -3)) 
		{
			tx[i1][j1] = 1;			ty[i1][j1] = 1;
			tx[i1][j1 + 1] = 1;		ty[i1][j1 + 1] = 1;
			tx[i1 + 1][j1] = 1;		ty[i1 + 1][j1] = 1;
			tx[i1 + 1][j1 + 1] = 1;	ty[i1 + 1][j1 + 1] = 1;
		}
	}
}
// gan tat ca cac diem nam giua (x1,y1) va (x2,y2) deu la diem noi 3//
void point3_point3(int **label, int y1, int x1, int y2, int x2) 
{
	int i, j;
	int minx = min(x1,x2); int maxx = max (x1,x2);
	int miny = min(y1,y2); int maxy = max (y1,y2);

	for (i = miny; i <= maxy; i++)
		for (j = minx; j <= maxx; j++)		
			if (0 == label[i][j])	label[i][j] = -3;			
	
	for (i = miny; i <= maxy; i++)
		for (j = minx; j <= maxx; j++)
		{			
			if ((label[i][j] == -3) && (Neighbor_3(label, i, j) == 0))				
				{					
					if (0 == label[i-1][j]) label[i][j] = label[i-1][j];
					if (0 == label[i+1][j]) label[i][j] = label[i+1][j];
					if (0 == label[i][j-1]) label[i][j] = label[i][j-1];
					if (0 == label[i][j+1]) label[i][j] = label[i][j+1];					
					label[y1][x1] = -3;label[y2][x2] = -3;										
				}
		}	
}

/*****************************************************************************/
/** line : Danh dau cac diem thuoc cung 1 duong 
 * @param CvPoint *point
 * [in]: Toa do cac diem anh.
 * @param t
 * [in]: So luong diem anh
 * @param label
 * [in_out]: Nhan cua diem anh 
 * @param status
 * [in_out]: Trang thai cua diem anh 
 * @param sumlabel
 * [out]: so luong duong
 * @return: none  
 *****************************************************************************/
void line(CvPoint *point, int t, int **label, int **status, int &sumlabel) 
{
	int k, i, j;	
	label[point[0].y][point[0].x] = 1;
	for (k = 1; k < t; k++) {
		j = point[k].x;
		i = point[k].y;
		if (label[i][j] == 0)
			if ((label[i][j - 1] < 0) && (label[i - 1][j] < 0)) 
			{
				label[i][j] = sumlabel + 1;
				sumlabel = sumlabel + 1;
			} else if (label[i][j - 1] > 0)
				label[i][j] = label[i][j - 1];
			else if (label[i - 1][j] > 0)
				label[i][j] = label[i - 1][j];
			else if (label[i - 1][j + 1] > 0)
				label[i][j] = label[i - 1][j + 1];
	}

	for (k = 0; k < t; k++)
	{
		j = point[k].x;		i = point[k].y;
		if (label[i][j] > 0)
			if ((label[i][j] != label[i - 1][j]) && (label[i - 1][j] > 0))
			{
				connect_2line(point, t, label, status, i, j, i - 1, j);
				status[i][j] = 0;		status[i - 1][j] = 0;
			} 
			else 
				if ((label[i][j] != label[i][j - 1])&& (label[i][j - 1] > 0)) 
			{
				connect_2line(point, t, label, status, i, j, i, j - 1);
				status[i][j] = 0;		status[i][j - 1] = 0;
			}
	}	
}
/////////////////////////
/*****************************************************************************/
/** check_point: xac dinh trang thai cua cac diem anh.
 * @param CvPoint *point
 * [in]: Toa do cac diem anh.
 * @param t
 * [in]: So luong diem anh
 * @param label
 * [in_out]: Nhan cua diem anh, cac diem can xoa thi nhan bang -1. 
 * @param status
 * [in_out]: Trang thai cua cac diem anh
 * @param sumlabel
 * [in_out]: so luong duong sau ket noi
 * @return: none   
 *****************************************************************************/
void check_point(CvPoint *point, int t, int **label, int **status, int &sumlabel) 
{
	int sum, k, i, j, t1, l, d1, d2, d;
	int minx, miny, maxx, maxy;

	CvPoint *point1;
	point1 = new CvPoint[t];
	l = 1;
	for (sum = 1; sum <= sumlabel; sum++) 
	{
		t1 = 0;
		minx = 10000;	miny = 10000;
		maxx = 0;		maxy = 0;
		d = 0;
		for (k = 0; k < t; k++) 
		{
			j = point[k].x;		i = point[k].y;
			if (sum == label[i][j]) 
			{
				point1[t1] = point[k];
				if (minx > point1[t1].x)	minx = point1[t1].x;
				if (miny > point1[t1].y)	miny = point1[t1].y;
				if (maxx < point1[t1].x)	maxx = point1[t1].x;
				if (maxy < point1[t1].y)	maxy = point1[t1].y;
				d1 = label_Neighbor(label, i, j, -3);
				d2 = label_Neighbor(label, i, j, -4);

				status[i][j] = 1; 
				if (d1 > 0)					status[i][j] = 0;
				if ((d1 == 0) && (d2 > 0))	status[i][j] = 0;	
				if ((d1 > 0) || (d2 > 0))	d = d + 1;
				t1 = t1 + 1;
				label[i][j] = l;
			}
		}
		if (t1 > 0) 
		{
			if ((maxx - minx <= threshold_delete_point)	&& (maxy - miny <= threshold_delete_point)) 
			{
				if (d <= 1)
					for (k = 0; k < t1; k++)
						label[point1[k].y][point1[k].x] = -1;									
				else
					l = l + 1;
			} else
				l = l + 1;
		}
	}
	sumlabel = l;
	delete[] point1;
}
/////////////////////////////////////////////////
/*****************************************************************************/
/**connect_2line: Ket noi giua 2 duong.
 * @param CvPoint *point
 * [in]: Toa do cac diem anh.
 * @param t
 * [in]: So luong diem anh 
 * @param x1,y1
 * [in]: Toa do diem mut duong 1
 * @param x2,y2
 * [in]: Toa do diem duong 2
 * @param label, status
 * [in_out]: Nhan cua diem anh, trang thai cua diem anh sau ket noi.
 * @return: none   
 *****************************************************************************/
void connect_2line(CvPoint *point, int t, int ** label, int **status, int y1, int x1, int y2, int x2) 
{
	int k, i, j, l;
	l = label[y1][x1];
	if ((label[y2][x2]>0)&& (l>0))   	
	{
		for (k = 0; k < t; k++) 
		{
			j = point[k].x;			i = point[k].y;
			if (label[i][j] == l)	label[i][j] = label[y2][x2];
		}
		status[y1][x1] = 1;	status[y2][x2] = 1;
	}	
}
////////////////////////////////////////////////////
/*****************************************************************************/
/**orientline: Tinh huong cua duong
 * @param CvPoint *point
 * [in]: Toa do cac diem anh.
 * @param t
 * [in]: So luong diem anh 
 * @param status
 * [in]: chi xet huong cua duong theo dau mut (status cua diem = 0)
 * @param label
 * [in]: Nhan cua diem anh, trang thai cua diem anh sau ket noi. 
 * @param orient
 * [in_out]: huong cua duong 
 * @param param1
 * [in]: So pixel tham gia vao tinh huong cua duong
 * @return: none   
 *****************************************************************************/
void orientline(CvPoint *point, int t,int **status, int **label, int **orient, int param1)
{
	int k,i,j,d1,d2;			
	for (k = 0; k < t; k++)
	{
		j = point[k].x; i = point[k].y;					 					
		if (0 == status[i][j])
		{
			orient[i][j]  = 181;
			find_orientline(label,orient,i,j,param1);
		}
	} 
}
void find_orientline(int **label,int **orient, int y, int x, int param1 )
{ 
	int label_xy = label[y][x];
	int i,j,t1=0;	
	int d = param1;
	if ((y<d)) d = y; if ((x<d)) d = x;
	for (i = y-d; i<=y+d; i++)
		for (j = x-d; j<=x+d; j++)              
			if ((label[i][j] == label_xy)&&(i>0)&&(j>0))
				if(orient_Neighbor(label,i,j,label_xy)==0)t1 = t1+1;	

	int miny = y, maxy = y, minx = x, maxx = x;			
	if (t1 > ((param1+5))) d = 5;		 	
	if (t1 < ( param1)) d = t1;	
	if ((y<d)) d = y; if ((x<d)) d = x;	

	for (i = y-d; i<=y+d; i++)
		for (j = x-d; j<=x+d; j++)              
			if ((label_xy == label[i][j]))					  
			{
				if (miny>i) miny = i;  if (maxy<i) maxy = i;
				if (minx>j) minx = j;  if (maxx<j) maxx = j;
			}	
	float d1, d2;
	if (y > miny) d1 = y - miny; else  d1 = y - maxy;
	if (x > minx) d2 = minx - x;  else  d2 = maxx - x;

	if (abs(d1) <= 2) orient[y][x] = 0;
	   else
	  	 if (abs(d2) <= 2) orient[y][x] = 90;							
		    else
			 orient[y][x] = cvRound(cvFastArctan(d1,d2));	
	if (orient[y][x] >= 180) orient[y][x] = orient[y][x] - 180;	
}
/// End_ orientline ///

/*****************************************************************************/
/**connect_4: Ket noi cac duong tai noi 4, cac duong tai noi 3 trong truong hop dac biet.
 * @param CvPoint *point
 * [in]: Toa do cac diem anh.
 * @param t
 * [in]: So luong diem anh 
 * @param label
 * [in_out]: Nhan cua diem anh sau ket noi.
 * @param status
 * [in]: chi xet cac duong chua ket noi.
 * @param orient
 * [in]: huong cua duong 
 * @param tx,ty
 * [in]: khoang cach hoanh do, tung do cua cac diem noi 3.
 * @param param2
 * [in]: Nguong de ket noi giua 2 duong
 * @return: none   
 *****************************************************************************/
void connect_4(CvPoint *point, int t, int **label, int **status, int **orient, int **tx, int **ty,int param2)
{
	int k, i,j;
	int threshold1 = 40, threshold2 = 150;
	if (param2 > threshold1) 
	{
		threshold1 = param2;
		threshold2 = 180 - param2;
	}
	for ( k =0; k<t; k++)
	{          
		i 	=  point[k].y;	j 	=  point[k].x;
		if ((label[i][j] == -4))
		{				
			if ((status[i+1][j] ==0)&&(status[i-1][j]== 0))					  
				connect_2line(point,t, label,status,i+1,j,i-1,j);				

			if ((status[i][j+1] ==0)&&(status[i][j-1]==0))									 
				connect_2line(point,t, label,status,i,j+1,i,j-1); 

		}   
		else						
			if ((label[i][j] == -3)&&(tx[i][j]==1)&&(ty[i][j]==1))
			{				
				if ((status[i][j-1] ==0)&&(status[i+1][j+2]== 0))
				  if ((abs(orient[i][j-1]- orient[i+1][j+2])<=threshold1)
					 ||(abs(orient[i][j-1]-orient[i+1][j+2])>=threshold2))
					connect_2line(point,t, label,status,i,j-1,i+1,j+2);				

				if ((status[i][j+1] ==0)&&(status[i+1][j-2]== 0))		      							
				  if ((abs(orient[i][j+1]- orient[i+1][j-2])<=threshold1)
				     ||(abs(orient[i][j+1]-orient[i+1][j-2])>=threshold2))
					connect_2line(point,t, label,status,i,j+1,i+1,j-2);

				if ((status[i-1][j] ==0)&&(status[i+2][j+1]==0))							
				  if ((abs(orient[i-1][j]- orient[i+2][j+1])<=threshold1)
				     ||(abs(orient[i-1][j]-orient[i+2][j+1])>=threshold2))
					connect_2line(point,t, label,status,i-1,j,i+2,j+1); 

				if ((status[i-1][j] ==0)&&(status[i+2][j-1]==0))						   
				  if ((abs(orient[i-1][j]- orient[i+2][j-1])<=threshold1)
					 ||(abs(orient[i-1][j]-orient[i+2][j-1])>=threshold2))
					connect_2line(point,t, label,status,i-1,j,i+2,j-1); 
			} 
			else
				if ((label[i][j] == -3)&& (tx[i][j]== 0)&& (ty[i][j] > 0))
				{	
					if ((status[i-1][j] ==0) && (status[i+1+ty[i][j]][j]== 0))	
						if ((abs(orient[i-1][j]- orient[i+1+ty[i][j]][j])<=threshold1)
						  ||(abs(orient[i-1][j]- orient[i+1+ty[i][j]][j])>=threshold2))
							connect_2line(point,t, label,status,i-1,j,i+1+ty[i][j],j);	

					if ((status[i+ty[i][j]][j+1]==0) && (status[i][j-1]==0))	
						if ((abs(orient[i+ty[i][j]][j+1]-orient[i][j-1])<=threshold1)
						  ||(abs(orient[i+ty[i][j]][j+1]-orient[i][j-1])>=threshold2))
							connect_2line(point,t, label,status,i+ty[i][j],j+1,i,j-1);																														

					if ((status[i+ty[i][j]][j-1] == 0) && (status[i][j+1]==0))										
						if ((abs(orient[i+ty[i][j]][j-1] - orient[i][j+1])<= threshold1)
						  ||(abs(orient[i+ty[i][j]][j-1] - orient[i][j+1])>=threshold2))
							connect_2line(point,t, label,status,i+ty[i][j],j-1,i,j+1);								

				}
				else
					if ((label[i][j] == -3)&& (ty[i][j] == 0)&& (tx[i][j] > 0))
					{
						if ((status[i][j+1+tx[i][j]] ==0)&&(status[i][j-1]==0))	
						    if ((abs(orient[i][j+1+tx[i][j]]-orient[i][j-1])<=threshold1)
							  ||(abs(orient[i][j+1+tx[i][j]]-orient[i][j-1])>=threshold2))
								connect_2line(point,t, label,status,i,j+1+tx[i][j],i,j-1); 							

						if ((status[i-1][j] ==0)&&(status[i+1][j+tx[i][j]]== 0))	
							if ((abs(orient[i-1][j]-orient[i+1][j+tx[i][j]])<=threshold1)
							  ||(abs(orient[i-1][j]-orient[i+1][j+tx[i][j]])>=threshold2))
								connect_2line(point,t, label,status,i-1,j,i+1,j+tx[i][j]);											

						if ((status[i-1][j] ==0)&&(status[i+1][j-tx[i][j]]==0))						
							if ((abs(orient[i-1][j]-orient[i+1][j-tx[i][j]])<=threshold1)
							  ||(abs(orient[i-1][j]-orient[i+1][j-tx[i][j]])>=threshold2))
								connect_2line(point,t, label,status,i-1,j,i+1,j-tx[i][j]);
					}	
	}
}
///////////////////
// begin connect_3//
/*****************************************************************************/
/**connect_3: Ket noi cac duong tai noi 3
* xac dinh cac duong phia trai, phia tren, phia duoi, phia phai cua cac diem noi 3
* voi cac duong cach no khoang tx, ty. Ket noi cac duong co huong gan nhau nhat 
 *****************************************************************************/
void connect_3(CvPoint *point, int t,int **tx, int **ty,int **orient, int **label, int **status, int param2)
{
	int k,i,j,d;	
	for (k = 0; k < t; k++)
	{
		j = point[k].x;		i = point[k].y;		        
		if ((label[i][j] == -3)&&(Neighbor_3(label,i,j)==0))			
		{					
			{				
				if (status[i][j-1] == 0)
				{
					d = 1;link3(point,t,orient,label,status,tx,ty,i,j,i,j-1,d,param2);	
				}	
				if (status[i-1][j] == 0) 
				{
					d = 2;link3(point,t,orient,label,status,tx,ty,i,j,i-1,j,d, param2);
				}	
				if (status[i+1][j] == 0)
				{
					d = 3;link3(point,t,orient,label,status,tx,ty,i,j,i+1,j,d,param2);							
				}	
				if (status[i][j+1] == 0)
				{
					d = 4;link3(point,t,orient,label,status,tx,ty,i,j,i,j+1,d,param2);							
				}
			}			
		}
	}
}
void link3(CvPoint *point,int t, int **orient, int **label, int **status,int **tx, int **ty, int i,int j,int y1, int x1,int k,int param2)
{
	int min = 181,kt=0;		
	status[y1][x1]=1;
	
	if (k!=1) find_min(point,t,orient,status,i+ty[i][j],j-1-tx[i][j],y1,x1,min);  	
	if (k!=2) find_min(point,t,orient,status,i+ty[i][j]-1,j+tx[i][j],y1,x1,min);
	if (k!=3) find_min(point,t,orient,status,i+1+ty[i][j],j+tx[i][j],y1,x1,min);	
	if (k!=4) find_min(point,t,orient,status,i+ty[i][j],j+1+tx[i][j],y1,x1,min);	
	
	if (k!=1) link(point,t,orient,label,status,i+ty[i][j],j-1-tx[i][j],y1,x1,min,param2,kt); 		
	if (k!=2) link(point,t,orient,label,status,i+ty[i][j]-1,j+tx[i][j],y1,x1,min,param2,kt);  	
	if (k!=3) link(point,t,orient,label,status,i+1+ty[i][j],j+tx[i][j],y1,x1,min,param2,kt); 	    
	if (k!=4) link(point,t,orient,label,status,i+ty[i][j],j+1+tx[i][j],y1,x1,min,param2,kt); 	

	if (kt == 0) status[y1][x1]=0;
}	
void find_min(CvPoint *point, int t, int **orient, int **status, int i, int j, int y, int x, int &min) 
{
	int dis_goc;
	if (status[i][j] == 0) 
	{
		dis_goc = abs(orient[y][x] - orient[i][j]);
		if (dis_goc >= 165)		dis_goc = 180 - dis_goc; 
		if (min > dis_goc)		min = dis_goc;
	}	
}
void link (CvPoint *point,int t, int **orient, int **label, int **status, int i,int j,int y, int x,int min, int param2, int &kt)
{		
	const int d = param2;		
	if (status[i][j] == 0)
		if ((abs(orient [y][x] - orient[i][j]) == min)||(abs(orient [y][x] - orient[i][j]) == (180 - min)))		
			if (min <= d) { connect_2line(point,t,label,status,i,j,y,x); kt = kt+1; }		
}
// end connect_3///

//*****************************************************************************/
/**connect_gd: Ket noi cac duong tai noi 3 duoc coi la doan tiep tuyen
 *****************************************************************************/
void connect_gd(CvPoint *point, int &t,int **tx, int **ty,int **orient, int **label, int **status, int sumlabel,int param2)
{	
	int k,i,j,d, t_3,t_gd, kt=0, sum_gd = 0;;		
	int dk1 =0, dk2 = 0;

	CvPoint *point_3;
	point_3 = new CvPoint[t];
	CvPoint *point_gd;
	point_gd = new CvPoint[t];
	vector<CvPoint> list_point;	

	for (int sum = 1; sum <= sumlabel; sum++ )	
	{
		t_gd = 0; d=0, sum_gd = 0;
		for (k = 0; k < t; k++)
		{
			j = point[k].x;		i = point[k].y;		        
			if (sum == label[i][j])
			{
			  point_gd[t_gd] = point[k];
			  t_gd = t_gd + 1;
			  if (1 == label_Neighbor(label,i,j,-3))// dau mut cua line
			  {
				  list_point.push_back(point[k]);			
				  d = d+1;				
			  }	
			  if(orient_Neighbor(label,i,j,sum)==0)
					sum_gd = sum_gd + 1;
			}
		}		
		if( 2 == d)// duong co 2 dau noi voi giao diem
		{				
			CvPoint point1, point2;
			point1 = list_point.at(0);
			point2 = list_point.at(1);
			float threshold_check_line = 1.5;			
			
			if ((CheckLine(point1,point2,sum_gd,threshold_check_line) == 1)
				&& ((abs(orient[point1.y][point1.x] - orient[point2.y][point2.x]) <= 10)
				  ||(abs(orient[point1.y][point1.x] + orient[point2.y][point2.x] -180) <= 10)))				  
			{			
				t_3= 0; dk1 = 0; dk2 = 0;
				double hsa,hsb,hsc;			
				CvPoint tmp;
				tmp.x = point1.x - point2.x;  tmp.y = point1.y - point2.y;
				hsa= -tmp.y;	hsb= tmp.x;	hsc = (-1)*hsa*point1.x - hsb*point1.y;				
				// xac dinh noi 3//
				for (int y = -1; y <= 1; y++)
					for (int x = -1; x <= 1; x++)
						if (1 == (abs(x)+ abs(y)))	
						{
							if (-3 == label[list_point.at(0).y + y][list_point.at(0).x + x])				
							{
								point1.y = list_point.at(0).y + y; point1.x = list_point.at(0).x + x; 
							}
							if (-3 == label[list_point.at(1).y + y][list_point.at(1).x + x])				
							{
								point2.y = list_point.at(1).y + y; point2.x = list_point.at(1).x + x; 
							}
						}
				// end///
				label[list_point.at(0).y][list_point.at(0).x] = 0;//de khong tinh duong nay
				label[list_point.at(1).y][list_point.at(1).x] = 0;//de khong tinh duong nay


				if (label[point1.y -1 ][point1.x]>0)
				{ point_3[t_3].x = point1.x;  point_3[t_3].y = point1.y - 1; t_3 = t_3+1;}				
				if (label[point1.y +1 ][point1.x]>0)
				{ point_3[t_3].x = point1.x;  point_3[t_3].y = point1.y + 1; t_3 = t_3+1;}				
				if (label[point1.y][point1.x -1]>0)
				{ point_3[t_3].x = point1.x - 1;  point_3[t_3].y = point1.y; t_3 = t_3+1;}				
				if (label[point1.y][point1.x + 1]>0)
				{ point_3[t_3].x = point1.x + 1;  point_3[t_3].y = point1.y; t_3 = t_3+1;}
				
				if (2 == t_3)
				{	
					if  (((abs(orient[point_3[0].y][point_3[0].x] - orient[list_point.at(0).y][list_point.at(0).x])<= param2)
						||(abs(orient[point_3[0].y][point_3[0].x] + orient[list_point.at(0).y][list_point.at(0).x] - 180)<= param2 ))
					   &&((abs(orient[point_3[1].y][point_3[1].x] - orient[list_point.at(0).y][list_point.at(0).x])<= param2)
						||(abs(orient[point_3[1].y][point_3[1].x] + orient[list_point.at(0).y][list_point.at(0).x] - 180)<= param2)))
					{						
						dk1 = 1;
					}
				}			 				

				if (label[point2.y -1 ][point2.x]>0)
				{ point_3[t_3].x = point2.x;  point_3[t_3].y = point2.y - 1; t_3 = t_3+1;}				
				if (label[point2.y +1 ][point2.x]>0)
				{ point_3[t_3].x = point2.x;  point_3[t_3].y = point2.y + 1; t_3 = t_3+1;}				
				if (label[point2.y][point2.x -1]>0)
				{ point_3[t_3].x = point2.x - 1;  point_3[t_3].y = point2.y; t_3 = t_3+1;}				
				if (label[point2.y][point2.x + 1]>0)
				{ point_3[t_3].x = point2.x + 1;  point_3[t_3].y = point2.y; t_3 = t_3+1;}		
	
				if (4 == t_3)				
				{									
					if  (((abs(orient[point_3[2].y][point_3[2].x] - orient[list_point.at(1).y][list_point.at(1).x])<= param2)
						||(abs(orient[point_3[2].y][point_3[2].x] + orient[list_point.at(1).y][list_point.at(1).x] - 180)<= param2))
					   &&((abs(orient[point_3[3].y][point_3[3].x] - orient[list_point.at(1).y][list_point.at(1).x])<= param2)
						||(abs(orient[point_3[3].y][point_3[3].x] + orient[list_point.at(1).y][list_point.at(1).x] - 180)<= param2)))
					{						
						dk2 = 1;
					}
				}

				label[list_point.at(0).y][list_point.at(0).x] = sum;
				label[list_point.at(1).y][list_point.at(1).x] = sum;				
								
				float kc[4];
				kt = 0;						
			
				if ((4 == t_3)&&(1 == dk1)||(1 == dk2))														
				{						
				  int kt1  = 0, kt2 = 0; 
				  for ( i = 0; i < 4; i++)
					{			
						int cs_y = point_3[i].y, cs_x = point_3[i].x;
						int kt1 = 0;				

						for (int y = -5; y < 0 ; y++)
						{
							for (int x = -5; x <= 5 ; x++)
							{						
								if (label[point_3[i].y + y][point_3[i].x + x] == label[point_3[i].y][point_3[i].x])
								{									
									cs_y = point_3[i].y + y;
									cs_x = point_3[i].x + x;
									kt1 = 1;
									break;
								}
							if (1 == kt1) break;
							}
						}
						
						if ((cs_y == point_3[i].y)&&(cs_x == point_3[i].x))
						{							
						  for (int y = 1; y <= 5 ; y++)
							for (int x = -5; x <= 5 ; x++)
							{								
								if (label[point_3[i].y + y][point_3[i].x + x] == label[point_3[i].y][point_3[i].x])								
								{
									cs_y = point_3[i].y + y;
									cs_x = point_3[i].x + x;									
								}
							}
						}						
						kc[i] = ((hsa)* cs_x  + (hsb)* cs_y + hsc)/sqrt((hsa * hsa)+(hsb * hsb));		
						if (kc[i] < 0) kt1 = kt1+1;
						if (kc[i] > 0) kt2 = kt2+1;						
						
					}				

				if ((kt1 < 3)&&(kt2 < 3))
				{					
					for ( i=0; i< 3; i++)
					{
						kt = 0;
						if (0 == status[point_3[i].y][point_3[i].x])
						{
						  for (j = i+1; j<4; j++)						
							if ((kc[i]*kc[j] <= 0) && ((abs(point_3[i].y - point_3[j].y) > 1)||(abs(point_3[i].x - point_3[j].x)>1)))							
							{								
							 if ((abs(kc[i]+ kc[j]) < 0.2))							
							  {						
							   if (0 == status[point_3[j].y][point_3[j].x])
							   {
									connect_2line(point,t,label,status,point_3[i].y,point_3[i].x,point_3[j].y,point_3[j].x);					
									status[point_3[i].y][point_3[i].x] = 3;
									status[point_3[j].y][point_3[j].x] = 3;										 							
									kt = 1;
									break;
							   }
							  }
							}
						}
						if (1 == kt)
							{
								for (j = 0; j < t_gd; j++)
									label[point_gd[j].y][point_gd[j].x] = label[point_3[i].y][point_3[i].x];						
								status[list_point.at(0).y][list_point.at(0).x] = 3;
								status[list_point.at(1).y][list_point.at(1).x] = 3;
							}
					}
					for ( i=0; i< 3; i++)
					{
						kt =0;
						if (0 == status[point_3[i].y][point_3[i].x])
						{
						for (j = i+1; j<4; j++)
							if ((kc[i]*kc[j]>0)&&((abs(point_3[i].y - point_3[j].y) > 1)||(abs(point_3[i].x - point_3[j].x)> 1)))
							{						
								if (0 == status[point_3[j].y][point_3[j].x])
								  {
									connect_2line(point,t,label,status,point_3[i].y,point_3[i].x,point_3[j].y,point_3[j].x);			
									status[point_3[i].y][point_3[i].x] = 3;
									status[point_3[j].y][point_3[j].x] = 3;							 								
									kt = 1;
									break;
								  }
							}						
						}
						if (1 == kt)
						{
						  for (j = 0; j < t_gd; j++)
							label[point_gd[j].y][point_gd[j].x] = label[point_3[i].y][point_3[i].x];						
						  status[list_point.at(0).y][list_point.at(0).x] = 3;
						  status[list_point.at(1).y][list_point.at(1).x] = 3;
						}
					}// end for						
				}// end if			
			}
	
			}						
		}
		if (list_point.size()>0)list_point.clear();
	}	
	delete []point_3;
	delete []point_gd;	
}
///////////////////////////////////////////////////////
//*****************************************************************************/
/**connect_other: Ket noi cac duong tai trong cac truong hop con lai
 *****************************************************************************/
void connect_other(CvPoint *point, int t, int **label, int **status,int **orient, int param2) 
{
	int k, k1, i, j, i1, j1, kt = 1;
	int angle = (int)(param2/2);
	for (k = 0; k < t - 1; k++) 
	{
		i = point[k].y;		j = point[k].x;
		if (0 == status[i][j]) 
		{
			for (k1 = k + 1; k1 < t; k1++) 
			{
				i1 = point[k1].y;	j1 = point[k1].x;
				if (0 == status[i1][j1]) 
				{
					float kc = sqrt(float(i - i1) * float(i - i1)
								  + float(j - j1) * float(j - j1));				
					if (kc <= 10) 
						if ((label_Neighbor(label, i1, j1, -4) == 1)
						 || (label_Neighbor(label, i1, j1, -3) == 1)) 
						{							
							if ((abs(orient[i][j] - orient[i1][j1]) <= angle)
							  ||(abs(orient[i][j] - orient[i1][j1]) >= (180 - angle))) 
							{
								kt = 1;
								if ((abs(j - j1) <= 3) && (orient[i1][j1] <= 5)
										|| (abs(i - i1) <= 3)&& (abs(90 - orient[i1][j1])<= 5))
									kt = 0;
								if (1 == kt) 
								{
									connect_2line(point, t, label, status, i, j,i1, j1);
									break;
								}
							}
						}					
				}
			}
		}
	}
}
//////////////////////////////////////
//*****************************************************************************/
/**connect_31: Ket noi cac duong tai noi 3 tx=0, ty=0
 *****************************************************************************/
void connect_31(CvPoint *point, int t,int **tx, int **ty,int **orient, int **label, int **xet, int param2)
{
	int k,i,j;	
	CvPoint *point1;
	point1 = new CvPoint[t];	
	int t1=0; int goc1, goc2, goc3;
	for (k = 0; k < t; k++)
     {
       j = point[k].x;   i = point[k].y;		        
	   if ((label[i][j] == -3)&&(Neighbor_3(label,i,j)==0))
       {
		   t1 = 0;
		   if ((tx[i][j]==0)&&(ty[i][j]==0))
		   { 		   
			   if (xet[i+1][j] == 0) { point1[t1].x = j;   point1[t1].y = i+1; t1 = t1+1;}
			   if (xet[i][j+1] == 0) { point1[t1].x = j+1; point1[t1].y = i;   t1 = t1+1;}
			   if (xet[i][j-1] == 0) { point1[t1].x = j-1; point1[t1].y = i;   t1 = t1+1;}				   
			   if (xet[i-1][j] == 0) { point1[t1].x = j;   point1[t1].y = i-1; t1 = t1+1;}
			if (t1>2)
			{		
				if ( orient[point1[0].y][point1[0].x] > 165)
					 orient[point1[0].y][point1[0].x] = 180 - orient[point1[0].y][point1[0].x];
				
				if ( orient[point1[1].y][point1[1].x] > 165) 
					 orient[point1[1].y][point1[1].x] = 180 - orient[point1[1].y][point1[1].x];
				
				if ( orient[point1[2].y][point1[2].x] > 165) 
					 orient[point1[2].y][point1[2].x] = 180 - orient[point1[2].y][point1[2].x];

			   int a = abs(orient[point1[0].y][point1[0].x] - orient[point1[1].y][point1[1].x]);
			   int b = abs(orient[point1[0].y][point1[0].x] - orient[point1[2].y][point1[2].x]);
			   int c = abs(orient[point1[2].y][point1[2].x] - orient[point1[1].y][point1[1].x]);
				
			   int min_orient= a;
			   if (min_orient>b) min_orient = b;
			   if (min_orient>c) min_orient = c;	
			   if (min_orient <= param2)
			   {
				   if (a == min_orient) connect_2line(point,t,label,xet,point1[1].y,point1[1].x,point1[0].y,point1[0].x);
				   else
					   if (b == min_orient) connect_2line(point,t,label,xet,point1[2].y,point1[2].x,point1[0].y,point1[0].x);
					   else
						if (c == min_orient) connect_2line(point,t,label,xet,point1[2].y,point1[2].x,point1[1].y,point1[1].x);				   
			   }
			}
		   }				
	   }
		}
	delete []point1;
}
//*****************************************************************************/
/**connect_line: Ket noi cac duong 
 *****************************************************************************/
void connect_line(CvPoint *point, int t, int **label,int **status,int **orient,int **tx, int **ty,int param2, int &sumlabel)
{
	connect_gd(point,t,tx,ty,orient,label,status,sumlabel,param2); 
	connect_4(point,t,label,status,orient,tx,ty, param2);	
	connect_31(point,t,tx,ty,orient,label,status,param2); 
	connect_3(point,t,tx,ty,orient,label,status,param2); 
	connect_other(point,t,label,status,orient,param2);
}
//////////////////////////////////////
//*****************************************************************************/
/**draw: Ve cac duong, xac dinh so duong
 *****************************************************************************/
void draw(IplImage *img, CvPoint *point, int t, int **label, int &sumlabel) {
	int r = 255, g = 0, b = 0, t1 = 0, l = 1, sum, k, i, j;
	for (sum = 1; sum <= sumlabel; sum++) 
	{
		t1 = 0;
		for (k = 0; k < t; k++) 
		{
			j = point[k].x;		i = point[k].y;
			if (label[i][j] == sum) 
			{
				cvLine(img,point[k],point[k],CV_RGB(r,g,b),1);
				t1 = t1 + 1;	
				label[i][j] = l;
			}			
		}
		if (t1 > 0)
		{		
				l = l + 1;	
			if (r<0)   r = 255;	else  r = r-35;
			if (g>255) g = 5;	else  g = g+55;
			if (b>255) b = 10;	else  b = b+25;	
		}
#ifdef SHOW_IMAGE_TO_DEBUG
        cvShowImage("show line", img);
		cvWaitKey();
#endif
	}
#ifdef SHOW_IMAGE_TO_DEBUG
    //cvSaveImage("img_draw.jpg",img);	
	cvShowImage("show line", img);
	cvWaitKey();
#endif
	sumlabel = l - 1;
}
///////////////////////////////
/*****************************************************************************/
/**connect: Khep kin hinh.
 * @param    img
 * [in_out]: Anh
 * @param CvPoint *point
 * [in]: Toa do cac diem anh.
 * @param t
 * [in]: So luong diem anh 
 * @param label
 * [in]: Nhan cua diem anh sau ket noi.
 * @param status
 * [in]: chi xet cac duong chua ket noi.
 * @param c
 * [out]: duong thang
 * @param check
 * [out]: 0: la hinh khep kin, 1: la duong thang 
 * @return: none   
 *****************************************************************************/
void connect(IplImage *img, CvPoint *point, int t, int **label, int **status, 
             Line &c, int &check, CvSeq* lcontour)
{
	int k,i,j;	
	uchar* data= (uchar*)img->imageData;	 
	CvPoint *point1; point1 = new CvPoint[t];
	int t1 = 0, kt = 0;

	for (i = 0; i< img->height; i++)
		for (j = 0; j< img->width; j++)
			img->imageData[i*img->widthStep+j] = (uchar)0;// line->point		
	for (k = 0; k < t; k++)
	{
		j = point[k].x; i = point[k].y;	kt = 0;
		if (3 == status[i][j])// doan noi cua tiep tuyen hoac cat nhau dai
		{ 		  
		  kt =1;
		}
		else 
		{
			if (label_Neighbor(label,i,j,label[i][j])==1) 
			{ 			
				status[i][j] = 0; kt =1; 
			}
		    if ((label_Neighbor(label,i,j,-3)==1)||((label_Neighbor(label,i,j,-4))==1))
			{				
				status[i][j] = 2;  kt =1;
			}
		}
////////////////////////////
        CvPoint* p = new CvPoint;
        p->x = j;
        p->y = i;
        cvSeqPush( lcontour,(CvPoint*)p);
        printf("\n(%d, %d)", p->x, p->y);
        if(p->x == 414)
            printf("debug 1");
        
/////////////////////////////////
		img->imageData[i*img->widthStep+j] = (uchar)255;// line->point	

		if (1 == kt)
		{
			point1[t1] = point[k]; t1 = t1+1;
		}		
	}	
	CvPoint p[2];	
	int t_point = 0;	
	connect_point(img,point1,t1,t,status, lcontour);				
	
	for (k = 0; k < t1; k++)
		if (0 == status[point1[k].y][point1[k].x])
		{
			p[t_point] = point1[k];
			t_point = t_point + 1;			
		}
		if (2 == t_point) 
		{
			c.begin.x = p[0].x;c.begin.y = p[0].y;
			c.end.x = p[1].x;c.end.y = p[1].y;			
			check = 1;
		}
	delete []point1;	
}
void connect_point(IplImage *img, CvPoint *point, int t, int t_label,int **status, CvSeq* lcontour) 
{
	int pixel_other = 30;
	int k1,k2,kt=0;		
	for (k1 = 0; k1 < t; k1++)
	{			
	  if (3 == status[point[k1].y][point[k1].x])
	    for (k2 = k1+1; k2 < t; k2++)
		{
		  if (3 == status[point[k2].y][point[k2].x])
		  {		    
			  int kc =(int) sqrt(float(point[k2].x - point[k1].x)* float(point[k2].x - point[k1].x)
				          + float(point[k2].y - point[k1].y)* float(point[k2].y - point[k1].y));
			  if (kc>1)
			      t_label = t_label + kc;

//////////////////
                    CvPoint* p = new CvPoint;
                    p->x = point[k2].x;
                    p->y = point[k2].y;
                    cvSeqPush( lcontour,(CvPoint*)p);
                    printf("\n(%d, %d)", p->x, p->y);
                    if(p->x == 256)
                    printf("debug 2");

///////////////////////////////
			  cvLine(img,point[k1],point[k2],CV_RGB(255,255,255),1,CV_AA);						
			  status[point[k1].y][point[k1].x] =1; status[point[k2].y][point[k2].x] =1;
			  break;
		  }
		}	
	}	

	for (k1 = 0; k1 < t; k1++)
	{			
	  if (2 == status[point[k1].y][point[k1].x])
	    for (k2 = k1+1; k2 < t; k2++)
		{
		  if (2 == status[point[k2].y][point[k2].x])
		  {												
		    if ((abs(point[k1].y - point[k2].y)<=10)&&(abs(point[k1].x - point[k2].x)<=10))				
			{
//////////////////////
           
                CvPoint* p = new CvPoint;
                p->x = point[k2].x;
                p->y = point[k2].y;
                cvSeqPush( lcontour,(CvPoint*)p);
                printf("\n(%d, %d)", p->x, p->y);
                if(p->x == 256)
                {
                    printf("debug 3");
                    cvLine(img,point[k1],point[k2],CV_RGB(255,255,255),1,CV_AA);						
                    cvCircle(img, point[k1], 3, CV_RGB(0,0,255), 2);
                    cvCircle(img, point[k2], 3, CV_RGB(0,0,255), 2);
                }
////////////////////////
			  cvLine(img,point[k1],point[k2],CV_RGB(255,255,255),1,CV_AA);						
			  status[point[k1].y][point[k1].x] =1; status[point[k2].y][point[k2].x] =1;
			  break;
			}									
		  }
		}	
	}

	for (k1 = 0; k1 < t; k1++)
	{		
	  if (2 == status[point[k1].y][point[k1].x])
		for (k2 = k1+1; k2 < t; k2++)
		{		
		  if (2 == status[point[k2].y][point[k2].x])
		  {			
			  if ((abs(point[k1].y - point[k2].y) <= pixel_other)&&
			 	(abs(point[k1].x - point[k2].x) <= pixel_other))
			 //  if ((abs(point[k1].y - point[k2].y) < t_label)&&
			//	(abs(point[k1].x - point[k2].x) < t_label))
				{
//////////////////////
                 
                    CvPoint* p = new CvPoint;
                    p->x = point[k2].x;
                    p->y = point[k2].y;
                    cvSeqPush( lcontour,(CvPoint*)p);
                    printf("\n(%d, %d)", p->x, p->y);
                    if(p->x == 256)
                        printf("debug 4");
////////////////////////
				    cvLine(img,point[k1],point[k2],CV_RGB(255,255,255),1,CV_AA);			
				    status[point[k1].y][point[k1].x] = 1; status[point[k2].y][point[k2].x] = 1;
				    break;
				}
		  }
	    }
	}

	for (k1 = 0; k1 < t; k1++)
	  if (2 == status[point[k1].y][point[k1].x]) status[point[k1].y][point[k1].x] = 0;				

	for (k1 = 0; k1 < t-1; k1++)
	{	
	  if (0 == status[point[k1].y][point[k1].x])
		for (k2 = k1+1; k2 < t; k2++)
	 	  {	
		    if (0 == status[point[k2].y][point[k2].x])
			  {				  
			    float threshold_check_line = 1.8;											
				if (CheckLine(point[k1],point[k2],t_label,threshold_check_line) == 0)					
				  {						
					int kc = sqrt(float(point[k1].x - point[k2].x)* float(point[k1].x - point[k2].x)
								+ float(point[k1].y - point[k2].y)* float(point[k1].y - point[k2].y));						
					if ((int)kc* 1.7 > t_label)
						{break;}
					else 
					  {
//////////////////////
                     
                        CvPoint* p = new CvPoint;
                        p->x = point[k2].x;
                        p->y = point[k2].y;
                        cvSeqPush( lcontour,(CvPoint*)p);
                        printf("\n(%d, %d)", p->x, p->y);
                        if(p->x == 256)
                            printf("debug 5");
////////////////////////
						cvLine(img,point[k1],point[k2],CV_RGB(255,255,255),1,CV_AA);
						status[point[k1].y][point[k1].x] = 1; status[point[k2].y][point[k2].x] = 1;								
						break;
					  }
				}
			  }
		  }		
	}
}
///// end_ connect ///// 

/*****************************************************************************/
/**track_point: Ham tong hop
 * @param    img
 * [in]: Anh
 * @param param1
 * [in]: So pixel de tinh huong cua duong
 * @param param2
 * [in]: Goc max de noi 2 duong
 * @param rec
 * [out]: du lieu HCN
 * @param circle
 * [out]: du lieu hinh tron
 * @param elip
 * [out]: du lieu elip
 * @param A
 * [out]: du lieu duong thang
 * @param dott_line
 * [out]: du lieu net dut
 * @param rec_text
 * [out]: du lieu vung chu
    
 *****************************************************************************/
void track_point(IplImage *src,int param1, int param2, vector<RectangularList> &rec, vector<CircleList> &circle,
					vector<Ellip> &ellip, vector<Line> &A, vector<Line> &dott_line, vector<RectangularList> &rec_text)
{
	double ti1 = (double)cvGetTickCount();	
	double time1 = (double)cvGetTickCount();	
	int i,j, t=0;	
	/// Resize anh 1 chieu bang 640///
    /*
	int max_length = max(src->width,src->height);
	if (max_length>640)
	{
		float tile = (float)640/max_length;			
		int w = src->width*tile;
		int h = src->height*tile;	
		IplImage *des = cvCreateImage( cvSize(w,h),IPL_DEPTH_8U,1);		
		cvResize(src, des, CV_INTER_AREA);					
		src = cvCloneImage(des);
		cvReleaseImage(&des);
	}
    */
	/// End _resize ////
	/// Nhi phan va thinning anh //
	IplImage* img_bin = nhiphan(src);
    //cvSaveImage("binary.jpg",img_bin);
    /*cvDilate(img_bin, img_bin, NULL, 3);
    cvSaveImage("dilate.jpg",img_bin);
    cvErode(img_bin, img_bin, NULL, 3);
    cvSaveImage("erode.jpg",img_bin);*/
    


	Stentiford(img_bin);	
    double time0 = (double)cvGetTickCount() - time1;
    printf("stentiford:  %g ms\n",time0/(cvGetTickFrequency() * 1000.));	

	//cvSaveImage("thin.jpg",img_bin);
	IplImage *img = img_bin; //cvLoadImage("thin.jpg",1);
#ifdef SHOW_IMAGE_TO_DEBUG
	cvShowImage("thin", img_bin);
	cvWaitKey();
#endif
	/// khi nhan dang duong hay hinh nao se xoa tren img_vb
	IplImage *img_vb  = cvCloneImage(img_bin);  
	double time2 = (double)cvGetTickCount() - time1;
	printf("nhi phan_thinning:  %g ms\n",time2/(cvGetTickFrequency() * 1000.));	

	double time3 = (double)cvGetTickCount();		
	CvSize Ssize;
	Ssize.width = img_bin->width+50;
	Ssize.height = img_bin->height+50;	
	CvPoint *point;	
	point = new CvPoint[img_bin->width*img_bin->height];	
	int **label;   CreateIntMatrix(label ,Ssize);
	int **orient;  CreateIntMatrix(orient ,Ssize);         	
	int **status;  CreateIntMatrix(status ,Ssize);
	int **tx;      CreateIntMatrix(tx ,Ssize);
	int **ty;      CreateIntMatrix(ty ,Ssize);


	for (i = 0; i< img_bin->height+50; i++)
		for (j = 0; j< img_bin->width+50; j++)
		{
			label[i][j] = -2; status[i][j] = 1;
		}		
	int sumlabel = 1;   
	intersection(img_bin,point,t,label,tx,ty);		
	line(point,t,label,status,sumlabel);				
	check_point(point,t,label,status,sumlabel);					/// xac dinh trang thai cac point; danh dau point can xoa 
				
	for (i = 0; i< t; i++)
		if (label[point[i].y][point[i].x] == -1)								
			data(img_bin, point[i].x,point[i].y) = (uchar)0;	// xoa cac point thuoc line ngan				

		
	t = 0;	sumlabel = 1;
	intersection(img_bin,point,t,label,tx,ty);					/// tim cac diem giao
	line(point,t,label,status, sumlabel);						/// xac dinh cac duong 
	check_point(point,t,label,status,sumlabel);					/// xac dinh trang thai cac point 
	orientline(point,t,status,label,orient,param1);				/// xac dinh huong cua 1 line
	connect_line(point,t,label,status,orient,tx,ty,param2,sumlabel);///	ket noi cac line 						
	draw(img,point,t,label,sumlabel);							/// ve cac line	

	Line line_solid;
	vector<Line> LineA;
	

///////////////Nhan dang duong //////////////////////////////////				

    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lcontour = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );

	int t_label =0;
	CvPoint *point_label;
	point_label = new CvPoint[t];
	int minx = 10000, miny = 10000, maxx = 0, maxy = 0;	
	int check =0;

	for (int k=1; k<=sumlabel; k++)																					
		{	
				t_label = 0;
				minx = 10000; miny = 10000; maxx = 0; maxy = 0;	
				for (int k_label = 0; k_label < t; k_label++)
				{
					j = point[k_label].x; i = point[k_label].y;
					if (k == label[i][j])
					{
						point_label[t_label] = point[k_label];						
						if (minx > point[k_label].x) minx = point[k_label].x;
						if (miny > point[k_label].y) miny = point[k_label].y;
						if (maxx < point[k_label].x) maxx = point[k_label].x;
						if (maxy < point[k_label].y) maxy = point[k_label].y;
						////
						t_label = t_label+1;															
					}			
				}						
				if ((maxx - minx>=20)||(maxy - miny>=20))
				{							 
				  check = 0;
                  connect(img_bin,point_label,t_label,label,status,line_solid,check, lcontour);		
                  for( CvSeq* c = lcontour; c!=NULL; c=c->h_next ){
                        for( int i=1; i < c->total; i++ ){
			                CvPoint* p1 = CV_GET_SEQ_ELEM( CvPoint, c, i - 1);
                            CvPoint* p2 = CV_GET_SEQ_ELEM( CvPoint, c, i );
			                //printf("%d (%d,%d)\n",i+1, p->x, p->y );
                            cvLine(src, *p1, *p2, CV_RGB(255,0,0),1);
		                }
	                }
                  cvClearSeq(lcontour);
#ifdef SHOW_IMAGE_TO_DEBUG
                  /* cvShowImage("anhconnectseq",src);
                  cvShowImage("anhconnectbin",img_bin);
				  cvWaitKey();		*/	
                  //lcontour->
#endif
				  if (0 == check)
				  {					
					int kt = 0;							
					CvMemStorage* storage = cvCreateMemStorage();
					CvSeq* first_contour = NULL;							
					int Nc = cvFindContours(img_bin,storage,&first_contour,sizeof(CvContour),CV_RETR_LIST, CV_CHAIN_APPROX_NONE);														
					/// Nhan dang hinh co ban	
					if (Nc>=2)
					{										
						for( CvSeq* con=first_contour; con!=NULL; con=con->h_next )
						{										
							if(con->total >= 40)
							{
								RectangularRecognition(img_bin, con, rec,kt);
								if (kt==0)
								{
									CircleRecognition( img_bin, con, circle,kt);									
									if (kt==0) EllipseRecognition( img_bin, con,ellip,kt);
								}														
							}
							//if (kt>0)
							{										
								for (i=0; i<t_label; i++)									
									data(img_vb,point_label[i].x,point_label[i].y) = (uchar)0;
								if (kt>0)
								{
								cvClearSeq(con);
								break;
								}
							}								
						}
					}
							//cvReleaseMemStorage(&storage);
							cvClearMemStorage(storage);
				  }		
				  /// end//
				  else									    
					   solid_line(img_vb,point_label, t_label,line_solid,LineA);// nhan dang duong thang
				}
		}				

		for(int i = 0; i < LineA.size(); i++)
			cvLine(img, cvPoint(LineA[i].begin.x, LineA[i].begin.y), cvPoint(LineA[i].end.x, LineA[i].end.y), cvScalar(255, 255, 255), 8, 1);

#ifdef SHOW_IMAGE_TO_DEBUG
        cvShowImage("line", img);
		cvvWaitKey();
#endif
		////////////dotted_line//////////
		sumlabel =1;				
		dotted_line(img_vb,point, t,label, status,orient,tx, ty,sumlabel,dott_line);// xac dinh net dut										
		/////////// end_dotted_line ////

		/////////// arrow ///////////////		
		if (LineA.size()>0)
			arrow(LineA,point,t,label,orient,sumlabel);	// xac dinh mui ten					
		/////////// end_arrow ////

		double time4 = (double)cvGetTickCount() - time3;
		printf("detection:  %g ms\n",time4/(cvGetTickFrequency() * 1000.));

		connect_solid_line(LineA,A); // nan lai duong thang				
		//A = LineA;
		LineA.clear();
		rect_text(point,t,label, sumlabel,rec_text);// xac dinh vung chu		
		draw_out(src,rec,circle,ellip,A, dott_line, rec_text);// ve lai cac doi tuong hinh

		double ti2 = (double)cvGetTickCount() - ti1;
		printf("tongcong:  %g ms\n",ti2/(cvGetTickFrequency() * 1000.));

		////////////////////////////////////////////////////////////////////		
		///////// Release Memory ////////
		delete []point;	
		delete []point_label;			
	
		//cvSaveImage("img_ve.jpg", img);
		//cvReleaseImage(&img);				
		//cvReleaseImage(&img_bin);	
		//cvReleaseImage(&img_vb);	
		//ReleaseIntMatrix(tx,Ssize.height);
		//ReleaseIntMatrix(ty,Ssize.height);	
		//ReleaseIntMatrix(orient,Ssize.height);
		//ReleaseIntMatrix(status,Ssize.height);
		//ReleaseIntMatrix(label,Ssize.height);	

}
/*****************************************************************************/
/**draw_out: Ve lai cac duong va cac hinh
 *****************************************************************************/
void draw_out(IplImage *src,vector<RectangularList> &rec, vector<CircleList> &circle,
					vector<Ellip> &ellip, vector<Line> &solid, vector<Line> &dott_line, vector<RectangularList> &rec_text)
{
	 Line a;
	 RectangularList rc;
	 CircleList cr;
	 Ellip el;

	IplImage *img_text = cvCreateImage(cvSize(src->width, src->height),IPL_DEPTH_8U,3);	

	for (int k = 0; k < 3; k++)
	  for (int i = 0; i<img_text->height; i++)
 		 for (int j = 0; j<img_text->width; j++)
			{
				img_text->imageData[i*img_text->widthStep+j * img_text->nChannels + k] = (uchar)255;
			}


	IplImage *img_binary =  cvCreateImage(cvSize(src->width, src->height),IPL_DEPTH_8U,1);
	cvThreshold(src,img_binary,110,255,CV_THRESH_BINARY);	
		
	for (int i=0; i<rec_text.size(); i++)				
	{
		rc = rec_text.at(i);      
		CvRect rect = cvRect(rc.xmin, rc.ymin, rc.xmax - rc.xmin, rc.ymax- rc.ymin);
		cvSetImageROI(src, rect);         
		IplImage *img2 = cvCreateImage(cvSize(rc.xmax - rc.xmin, rc.ymax- rc.ymin),
								   img_text->depth, img_text->nChannels);
     
	    for (int k =0; k<3; k++)
		  for (int y = rc.ymin; y < rc.ymax; y++)
			for (int x = rc.xmin; x < rc.xmax; x++)
			{					
				img_text->imageData[y*img_text->widthStep+x * img_text->nChannels + k] = (uchar)(img_binary->imageData[y*img_binary->widthStep+x]);
			}
	}

	 for (int i=0; i<solid.size(); i++)
	 {
		 a = solid.at(i);
		 CvPoint q, q1;
		 q.x = a.begin.x; q.y = a.begin.y;
		 q1.x = a.end.x; q1.y = a.end.y;	
		  cvLine(img_text,q,q1,CV_RGB(0,0,0),1,CV_AA);

		  if (a.begin.arw == 1)
			 cvLine(img_text,q,q,CV_RGB(255,0,0),5);
		  if (a.end.arw == 1)	 
			 cvLine(img_text,q1,q1,CV_RGB(255,0,0),5);
	 }

	for (int i=0; i<dott_line.size(); i++)
	 {
		 a = dott_line.at(i);
		 CvPoint q, q1;
		 q.x = a.begin.x; q.y = a.begin.y;
		 q1.x = a.end.x; q1.y = a.end.y;		 
		 cvLine(img_text,q,q1,CV_RGB(0,0,0),1,CV_AA);
	 }

	 for (int i=0; i<rec.size(); i++)
	 {
		rc = rec.at(i);	 
		cvRectangle(img_text, cvPoint(rc.xmin,rc.ymin), cvPoint(rc.xmax,rc.ymax),CV_RGB(0,0,0), 1);
	 }

	 for (int i=0; i<circle.size(); i++)
	 {
		cr = circle.at(i);	 
		cvCircle(img_text, cvPoint(cr.x_center,cr.y_center),cr.radius,CV_RGB(0,0,0),1,8);
	 }

	 for (int i=0; i<ellip.size(); i++)
	 {
		el = ellip.at(i);
		cvEllipse(img_text,cvPoint(el.x_center,el.y_center),cvSize(el.max_radius,el.min_radius),
		el.angle,0, 360, CV_RGB(0,0,0),1,8,0 );
	 }

	 //solid.clear();
	 //dott_line.clear();
	 //rec.clear();
	 //circle.clear();
	 //ellip.size();
	 
	 //cvSaveImage("out.jpg", img_text);
	 //cvShowImage("out", img_text);
	 cvReleaseImage(&img_text);
}
/*****************************************************************************/
/**solid_line: Xac dinh duong thang
 * @param  img
 * [in]: Anh
 * @param  point
 * [in]: Toa do cac diem anh thuoc duong c
 * @param  t
 * [in]: So luong cac diem anh thuoc duong c
 * @param  c
 * [in_out]: duong thang, toa do diem dau, diem cuoi.
 * @param  A
 * [out]: Tap cac duong thang 
 *****************************************************************************/
void solid_line(IplImage *img, CvPoint *point, int t, Line &c, vector<Line> &A) 
{
	int kt = 0, max_pos;
	Line d;
	cvpoint q, q1, tmp;
	double hsa, hsb, hsc, kc, max_distance;
	q = c.begin;
	q1 = c.end;
	if ((abs(q.x - q1.x) >= threshold_pixel_line)|| (abs(q.y - q1.y) >= threshold_pixel_line)) 
	{
		tmp.x = q1.x - q.x;
		tmp.y = q1.y - q.y;
		hsa = -tmp.y;		hsb = tmp.x;		hsc = (-1) * hsa * q.x - hsb * q.y;
		max_distance = 0;
		max_pos = 0;
		for (int i = 0; i < t; i++) 
		{			
			data(img,point[i].x,point[i].y) = (uchar)0; // Xoa duong thang trong anh img			
			kc = abs(((hsa) * point[i].x + (hsb) * point[i].y + hsc)/ sqrt((hsa * hsa) + (hsb * hsb)));
			if (max_distance < kc) 
			{
				max_distance = kc;
				max_pos = i;
			}
		}
		if (max_distance / t > 0.1) 
		{
			q.arw = 0;		tmp.arw = 0;		q1.arw = 0;
			tmp.x = point[max_pos].x;
			tmp.y = point[max_pos].y;

			d.begin = q; d.end = tmp;
			A.push_back(d);
			
			d.begin = tmp; d.end = q1;			
			A.push_back(d);
		}
		else 
		{
			c.begin.arw = 0;
			c.end.arw = 0;
			A.push_back(c);
		}
	}	
}
/*****************************************************************************/
/**connect_solid_line: Ket noi cac duong thang va nan lai duong
 * @param  A
 * [in]: Tap cac duong thang chua ket noi
 * @param  LineA
 * [out]: Tap cac duong thang da ket noi 
 *****************************************************************************/
void connect_solid_line(vector<Line> A,vector<Line> &LineA)
{
	Line line_a,line_b;
	int i,j;
	int minx,miny, maxx, maxy; 

	for (i=0; i<A.size(); i++)			
	{
		line_a = A.at(i);				
		cvpoint tmp_a;		
		tmp_a.x = line_a.end.x - line_a.begin.x;  
		tmp_a.y = line_a.end.y - line_a.begin.y;
		int orient_line1 = cvRound(cvFastArctan((float)abs(tmp_a.y),(float)abs(tmp_a.x)));		

		double kc1=0.0, kc2=0.0;		
		double hsa,hsb,hsc;	
		hsa = 0.0; hsb = 0.0; hsc = 0.0;				
		hsa = -tmp_a.y;	hsb = tmp_a.x;	hsc = (-1)* hsa * line_a.begin.x - hsb * line_a.begin.y;				
		
	   for(int j = i+1; j< A.size(); j++) 											
		{	
		  line_b = A.at(j);										 
		  if(line_b.begin.arw>=0)
		   {					  	
			  tmp_a.x = line_b.begin.x - line_b.end.x;  
			  tmp_a.y = line_b.begin.y - line_b.end.y;		
			  int orient_line2 = cvRound(cvFastArctan((float)abs(tmp_a.y),(float)abs(tmp_a.x)));	
			  if (abs(orient_line2 - orient_line1)<=5)
			   {				   
				   kc1 = abs(((hsa)* line_b.begin.x + (hsb)* line_b.begin.y + hsc)/sqrt((hsa * hsa)+(hsb * hsb)));
				   kc2 = abs(((hsa)* line_b.end.x + (hsb)* line_b.end.y + hsc)/sqrt((hsa * hsa)+(hsb * hsb)));					 
				 if (((int)kc1<=10)&&((int)kc2<=10))
				  {					  		  
					if ((orient_line1 > 80)&&(orient_line2 > 80))
					{
						if (  (min(line_a.begin.x,line_a.end.x) < max(line_b.begin.x,line_b.end.x)
							&& max(line_b.begin.x,line_b.end.x) < max(line_a.begin.x,line_a.end.x))
							||(min(line_b.begin.x,line_b.end.x) < max(line_a.begin.x,line_a.end.x)
							&& max(line_a.begin.x,line_a.end.x) < max(line_b.begin.x,line_b.end.x))
							||(distance_line(line_a, line_b) <= 50))						
						{		
							miny = min(min(line_a.begin.y,line_a.end.y),min(line_b.begin.y,line_b.end.y));
							maxy = max(max(line_a.begin.y,line_a.end.y),max(line_b.begin.y,line_b.end.y));
							line_a.begin.y = miny;
							line_a.end.y = maxy;
							///update///
							if (miny == line_a.end.y)
								line_a.begin.arw = line_a.end.arw;
							if (miny == line_b.begin.y)
								line_a.begin.arw = line_b.begin.arw;
							if (miny == line_b.end.y)
								line_a.begin.arw = line_b.end.arw;

							if (maxy == line_a.begin.y)
								line_a.end.arw = line_a.begin.arw;
							if (maxy == line_b.begin.y)
								line_a.end.arw = line_b.begin.arw;
							if (maxy == line_b.end.y)
								line_a.end.arw = line_b.end.arw;
							///end///
							line_a.begin.x = (int)(line_a.begin.x + line_a.end.x +line_b.begin.x + line_b.end.x)/4;
							line_a.end.x = line_a.begin.x;		
						 
							line_b.begin.arw = -1;
							A.at(j) = line_b;
						}
						//break;
					}

					if ((orient_line1 < 10)&&(orient_line2 < 10))
					{
						if (  (min(line_a.begin.x,line_a.end.x) < max(line_b.begin.x,line_b.end.x)
							&& max(line_b.begin.x,line_b.end.x) < max(line_a.begin.x,line_a.end.x))
							||(min(line_b.begin.x,line_b.end.x) < max(line_a.begin.x,line_a.end.x)
							&& max(line_a.begin.x,line_a.end.x) < max(line_b.begin.x,line_b.end.x))
						||(distance_line(line_a, line_b) <= 50))
						{							
							minx = min(min(line_a.begin.x,line_a.end.x),min(line_b.begin.x,line_b.end.x));							
							maxx = max(max(line_a.begin.x,line_a.end.x),max(line_b.begin.x,line_b.end.x));												
							line_a.begin.x = minx;
							line_a.end.x = maxx;
							///update///
							if (minx == line_a.end.x)
								line_a.begin.arw = line_a.end.arw;
							if (minx == line_b.begin.x)
								line_a.begin.arw = line_b.begin.arw;
							if (minx == line_b.end.x)
								line_a.begin.arw = line_b.end.arw;

							if (maxx == line_a.begin.x)
								line_a.end.arw = line_a.begin.arw;
							if (maxx == line_b.begin.x)
								line_a.end.arw = line_b.begin.arw;
							if (maxx == line_b.end.x)
								line_a.end.arw = line_b.end.arw;
							///end update///

							line_a.end.y = line_a.begin.y;							
							line_b.begin.arw = -1;
							A.at(j) = line_b;
						}
						//break;
					}
				 }
			  }
			}
		}
		A.at(i) = line_a;		
		if(line_a.begin.arw>=0)		
		{				
			LineA.push_back(line_a);
		}
	}
		/// doan nan lai duong //
	for (i = 0; i < LineA.size();i++ )
	{
		line_a = LineA.at(i);				
		{
			cvpoint tmp_a;
			tmp_a.x = line_a.end.x - line_a.begin.x;  
			tmp_a.y = line_a.end.y - line_a.begin.y;		
			int orient_line = cvRound(cvFastArctan((float)abs(tmp_a.y),(float)abs(tmp_a.x)));			
			if (orient_line > 80)
			{
				LineA.at(i).begin.x = (int)(line_a.begin.x + line_a.end.x)/2;
				LineA.at(i).end.x = LineA.at(i).end.x;				
			}
			if (orient_line < 5)
			{				
				LineA.at(i).begin.y = (int)(line_a.begin.y + line_a.end.y)/2;
				LineA.at(i).end.y = LineA.at(i).begin.y;
			}
		}				
	}			
}
/*****************************************************************************/
/**rect_text:  Xac dinh cac vung chu
 * @param  point
 * [in]: Toa do cac diem anh. 
 * @param  t
 * [in]: So luong cac diem anh 
 * @param  label
 * [in]: Nhan cua cac diem
 * @param  sumlabel
 * [in]: So luong duong
 * @param  rec_text
 * [out]: Tap cac vung chu.
 *****************************************************************************/
void rect_text(CvPoint *point, int t,int **label,int sumlabel,vector<RectangularList> &rec_text)
{
	vector<RectangularList> list_hcn;
	vector<RectangularList> list_hcns;
	RectangularList hcn,a,b;
	int sum,k,i,j, kt = 0, index = 0;
	int t_hcn = 0, kc, check = 0;	
	int minx, maxx, miny, maxy;
	double dien_tich, duong_cheo;
	for ( sum = 0; sum <= sumlabel; sum++)			
	{
		kt = 0; t_hcn = 0; check = 0;
		minx = 10000; maxx = 0; miny = 10000; maxy = 0;
		for ( k = 0; k < t; k++)
		{
			i = point[k].y; j = point[k].x;
			if (sum == label[i][j])
			{
				t_hcn = t_hcn +1;
				if (minx > j) minx = j;	if (maxx < j) maxx = j;
				if (miny > i) miny = i;	if (maxy < i) maxy = i;				
				if ((label_Neighbor(label,i,j,-3)==1)||((label_Neighbor(label,i,j,-4))==1))
					check = 1;
			}
		}	
		duong_cheo = sqrt(float(maxx - minx)* float(maxx - minx)
				        + float(maxy - miny)* float(maxy - miny));
		dien_tich = (maxx - minx)*(maxy - miny);
		float ti_le = (float)duong_cheo/dien_tich;					
	
		if (((1 == check)||(ti_le <= 0.5))&& (maxx - minx < 30)&& (maxy - miny < 30))				
		{			
			hcn.xmin = minx;
			hcn.ymin = miny;			
			hcn.xmax = maxx;
			hcn.ymax = maxy;	
			list_hcn.push_back(hcn);			
		}
	}		
	if (list_hcn.size()>0)
	{
		for (i = 0; i < list_hcn.size()-1; i++)
		{
			index = 0;			
			for (j = i+1; j < list_hcn.size(); j++)	
			{			
				kt = 0;													
				join2hcn(list_hcn.at(i),list_hcn.at(j),kt);			
				if (1 == kt) 
				{					
					index = index + 1;						
				}
			}					
			if ((index >= 2)&&(((list_hcn.at(i).xmax - list_hcn.at(i).xmin) >=3)
				&& ((list_hcn.at(i).ymax - list_hcn.at(i).ymin >=3))))			
			{		
				list_hcns.push_back(list_hcn.at(i));		
			}			
		}
	}

		for (i = 0; i < list_hcns.size(); i++)
		{			
			for (j = i+1; j < list_hcns.size(); j++)	
			{		
				kt = 0;
				join2hcn(list_hcns.at(i),list_hcns.at(j),kt);														
			}						
		}
	if (list_hcns.size()>0)
	{
		a = list_hcns.at(0);
		if (( a.ymax - a.ymin >=10)&&( a.xmax - a.xmin >=10))
			rec_text.push_back(a);
	}
	kt = 0;
	for (i = 1; i < list_hcns.size(); i++)
	{
		a = list_hcns.at(i); kt = 0;
		for (j = 0; j < rec_text.size(); j++)			
		{
			b = rec_text.at(j);			
			if(( a.xmax == b.xmax)&&( a.xmin == b.xmin)&&
				( a.ymax == b.ymax)&&( a.ymin == b.ymin)) 
			{
				kt = 1;
				break;
			}			
		}
		if(0 == kt)
			if (( a.ymax - a.ymin >=10)&&( a.xmax - a.xmin >=10))			
				rec_text.push_back(a);
	}
	
	list_hcn.clear();
	list_hcns.clear();	
}
/*****************************************************************************/
/**join2hcn: Ghep 2 HCN o gan nhau
 * @param  hcn1, hcn2
 * [in_out]: Du lieu 2 hcn de tinh khoang cach 
 * @param  kt
 * [out]: 1: ghep 2 hcn thanh 1; 0: khong ghep 
 *****************************************************************************/
void join2hcn( RectangularList &hcn1, RectangularList &hcn2, int &kt)
{	
	int dis = 1000;
	RectangularList cn1,cn2;
	if (hcn1.ymin <= hcn2.ymin) 
	{
		cn1 = hcn1; cn2 = hcn2;
	}
	else
	{
		cn1 = hcn2; cn2 = hcn1;	
	}
	
	int delta_x = min(min (abs(cn1.xmin - cn2.xmin),abs(cn1.xmin - cn2.xmax)),
					  min (abs(cn1.xmax - cn2.xmin),abs(cn1.xmax - cn2.xmax)));
	int delta_y = min(min (abs(cn1.ymin - cn2.ymin),abs(cn1.ymin - cn2.ymax)),
					  min (abs(cn1.ymax - cn2.ymin),abs(cn1.ymax - cn2.ymax)));
		
	if ((delta_x <= 25)&& (delta_y <= 7))
	{
		dis = max(delta_x, delta_y); 		
	}
	
	if (cn2.ymin <= cn1.ymax) 
	{		
		if ((cn2.xmin <= cn1.xmax) && (cn1.xmax <= cn2.xmax)
		  ||(cn2.xmin <= cn1.xmin) && (cn1.xmin <= cn2.xmax))
			dis = 0;	
		else
			if ((cn1.xmin <= cn2.xmax) && (cn2.xmax <= cn1.xmax)
			  ||(cn1.xmin <= cn2.xmin) && (cn2.xmin <= cn1.xmax))
				dis = 0;
	}
	
	if (dis <= 25)	
	{	
		cn1.xmin = min(hcn1.xmin, hcn2.xmin) ;
		cn1.ymin = min(hcn1.ymin, hcn2.ymin) ;
		cn1.xmax = max(hcn1.xmax, hcn2.xmax) ;
		cn1.ymax = max(hcn1.ymax, hcn2.ymax) ;		
		if (cn1.ymax - cn1.ymin <= 50)
		{			
			kt = 1;
			hcn1 = cn1;
			hcn2 = hcn1;					
		}
	}			
}
/*****************************************************************************/
/**arrow: Xac dinh mui ten o 2 dau  duong thang
 * @param  A
 * [in]: Tap cac duong thang
 * @param  point
 * [in]: Toa do cac diem anh
 * @param  t
 * [in]: So luong cac diem anh
 * @param  label
 * [in]: Nhan cua cac diem
 * @param  orient
 * [in]: huong cac diem
 * @param  sumlabel
 * [in]: So luong cac duong thang 
 *****************************************************************************/
void arrow(vector<Line> &A, CvPoint *point, int t,int ** label, int **orient, int sumlabel) 
{
	Line line1, line2;
	cvpoint point_begin, point_end;
	int Begin_Arr[100];
	int begin_arr;
	int End_Arr[100];
	int end_arr;
	int orient_line1;
	float dx, dy;
	for (int k = 0; k < A.size(); k++) 
	{
		line1 = A.at(k);
		point_begin = line1.begin;
		point_end = line1.end;

		dx = (float) abs(point_begin.x - point_end.x);
		dy = (float) abs(point_begin.y - point_end.y);
		orient_line1 = cvRound(cvFastArctan(dy, dx));

		begin_arr = 0;
		line2 = line1;
		line_arrow(label, point_begin.x, point_begin.y, Begin_Arr, begin_arr);
		if (begin_arr > 0)
			Check_arrow(point_begin, orient_line1, line1, point, t, label, Begin_Arr, begin_arr);
		

		end_arr = 0;
		line2.begin = point_end;
		line2.end = point_begin;		
		line_arrow(label, point_end.x, point_end.y, End_Arr, end_arr);
		if (end_arr > 0)
			Check_arrow(point_end, orient_line1, line2, point, t, label, End_Arr, end_arr);
		A.at(k).begin = point_begin;
		A.at(k).end = point_end;		
	}
}
/*****************************************************************************/
/**Check_arrow: Danh dau 2 dau cua duong thang co mui ten?
 * @param  p
 * [in_out]: Diem dau hoac cuoi cua duong thang 
 * @param  orient_line1
 * [in]: 
 * @param  line1
 * [in]: 
 * @param  point
 * [in]:  Toa do cac diem anh
 * @param  t
 * [in]: So luong diem anh
 * @param  label
 * [in]: Nhan cua cac diem 
 * @param  Line_Arr[10]
 * [out]: Mang luu tru cac duong
 * @param  sumlabel_arr
 * [out]: so luong duong 
 *****************************************************************************/
void Check_arrow(cvpoint &p,int orient_line1,Line line1,CvPoint *point,int t,int **label,int Line_Arr[10],int sumlabel_arr)
{	
	int minx, miny, maxx, maxy, kt,k,t_arrow;
	vector<cvpoint> line2;
	Line line3;	
	int check_arrow = 0,i,j;
	float threshold_line_arrow = 2.0;	
	cvpoint q, q1;
	CvPoint begin, end;
	CvPoint *p_arrow; p_arrow =new CvPoint[t];
	for (int sum=0; sum < sumlabel_arr; sum++)			
	{
		kt=0; t_arrow = 0; 
		minx = 10000; maxx=0; miny=10000;maxy =0;
		for ( k=0; k<t; k++)
		{
			i = point[k].y; j = point[k].x;
			if (label[i][j] == Line_Arr[sum])
			{
				p_arrow[t_arrow] = point[k]; 					 
				t_arrow = t_arrow +1;
				if (minx > j) minx = j;	if (maxx < j) maxx = j;
				if (miny > i) miny = i;	if (maxy < i) maxy = i;
				if (label_Neighbor(label,i,j,Line_Arr[sum])==1) 
				{
					q.x = point[k].x ; q.y = point[k].y ;
					line2.push_back(q);				
				}						
			}
		}			
		if (line2.size()>1)
		{			
			begin.x = line2.at(0).x;  begin.y = line2.at(0).y;
			end.x = line2.at(1).x;  end.y = line2.at(1).y;			
			if ((CheckLine(begin,end,t_arrow,threshold_line_arrow)== 1)&&(t_arrow<=30))
			{					
				q  = line2.at(0); q1 = line2.at(1); 
				
				if (orient_line1<=45)
				{
					if ((0 == abs(line2.at(0).y - line2.at(1).y)) &&(abs(p.x - line2.at(0).x)> abs(p.x - line2.at(1).x)))
					{											
						q	= line2.at(1); q1 = line2.at(0);
					}					
					if (abs(p.y - line2.at(0).y)> abs(p.y - line2.at(1).y))
					{					
						q = line2.at(1); q1 = line2.at(0);
					}
				}
				else 
				{
					if ((0 == abs(line2.at(0).x - line2.at(1).x)) &&(abs(p.y - line2.at(0).y)>abs(p.y - line2.at(1).y)))
					{				
						q = line2.at(1); q1 = line2.at(0);
					}					
					if (abs(p.x - line2.at(0).x)> abs(p.x - line2.at(1).x))
					{				
						q = line2.at(1); q1 = line2.at(0);
					}
				}						
					line3.begin = q; line3.end = q1;
					int goc= AngleOf2Lines( line1, line3);							
					if ((goc>=10)&&(goc<=45)) 
					{						
						kt =1;  					
					}
			}
			else
			  if ((maxx - minx<=30)&&(maxy - miny<=30)&&(maxx - minx>0)&&(maxy - miny>0)) //cu = 20
			  {				
				if (orient_line1<=45)
				 {
				   if ((maxx > line2.at(0).x)&&(maxx > line2.at(1).x)||(minx < line2.at(0).x)&&(minx < line2.at(1).x))					   
					   if ((p.y >= min(line2.at(0).y,line2.at(1).y)) && (p.y <= max(line2.at(0).y,line2.at(1).y)))
						{
							if (((maxx == line2.at(0).x)||(maxx == line2.at(1).x))&&
								((abs(p.x - maxx) < abs(p.x - minx))||((p.x >= minx) && (p.x <= maxx))))							
								kt =1;					
							if (((minx == line2.at(0).x)||(minx == line2.at(1).x))&&
								((abs(p.x - minx) < abs(p.x - maxx))||(p.x >= minx) && (p.x <= maxx)))
								kt =1;
						}
				 }	
				else 				
				  {
					if ((maxy > line2.at(0).y)&&(maxy > line2.at(1).y)||(miny < line2.at(0).y)&&(miny < line2.at(1).y))					  
					  if ((p.x >= min(line2.at(0).x,line2.at(1).x)) && (p.y <= max(line2.at(0).x,line2.at(1).x)))
						{							
							if (((maxy == line2.at(0).y)||(maxy == line2.at(1).y))&&
								((abs(p.y - maxy) < abs(p.y - miny))||((p.y >= miny) && (p.y <= maxy))))							
								kt =1;						
							if (((miny == line2.at(0).y)||(miny == line2.at(1).y))&&
								((abs(p.y - miny) < abs(p.y - maxy))||((p.y >= miny) && (p.y <= maxy))))							
								kt =1;
						}					
				  }
			  }			 
			if (1 == kt)
			{	
				p.arw =1;		
			}
		}
		  if (line2.size()>0) line2.clear();		
	}
	delete []p_arrow;	
	}
/*****************************************************************************/
/**line_arrow: Xac dinh duong gan 2 dau cua duong
 * @param  label
 * [in]: Nhan cua cac diem
 * @param  x,y
 * [in]: Toa do diem dau hoac cuoi duong thang
 * @param  Line_Arr[10]
 * [out]: Mang luu tru cac duong
 * @param  sumlabel_arr
 * [out]: so luong duong  
 *****************************************************************************/
void line_arrow( int ** label, int x, int y, int Line_Arr[10],int &sumlabel_arr) 
{
	// int 	threshold_pixel_arrow = 10;
	int thresh = 10;
	int i, j;
	int d1 = y - thresh, d2 = y + thresh;
	int d3 = x - thresh, d4 = x	+ thresh;	
	if (d1 < 0)		d1 = thresh - y;	
	if (d3 < 0)		d3 = thresh - x;	
	int dem = 0;
	for (i = d1; i <= d2; i++)
		for (j = d3; j <= d4; j++)
		{
			if (label[i][j] > 0) 
			{
				if (0 == sumlabel_arr) 
				{
					Line_Arr[sumlabel_arr] = label[i][j];
					sumlabel_arr = sumlabel_arr + 1;
				} 
				else 
				{
					int kt = 0;
					for (int k = 0; k < sumlabel_arr; k++)
						if (label[i][j] == Line_Arr[k]) 
						{
							kt = 1;
							break;
						}
					if (0 == kt) 
					{
						Line_Arr[sumlabel_arr] = label[i][j];
						sumlabel_arr = sumlabel_arr + 1;
					}
				}
			}
		}
}

/*****************************************************************************/
/**dotted_line: Tim net dut
 * @param  img
 * [in]: Anh da loai duong thang, cac hinh.
  * @param  point
 * [in_out]: Toa do cac diem anh con lai
  * @param  t
 * [in_out]: so luong diem anh
 * @param  label,status,orient,tx,ty
 * [in_out]: Thuoc tinh diem anh
 * @param  sumlabel
 * [in_out]: Tap cac duong
 * @param  dot_line
 * [out]: Tap cac net dut 
 *****************************************************************************/
void dotted_line(IplImage *img, CvPoint *point, int t, int **label,
		int ** status, int **orient, int **tx, int **ty,
		int &sumlabel, vector<Line> &dott_line) 
{
	float threshold_line_dotted = 1.3;
	Line c, d;
	vector<Line> A,B;	
	int i, j;	
	cvpoint q, q1;

	for (i = 0; i < img->height + 50; i++)
		for (j = 0; j < img->width + 50; j++) {
			label[i][j] = -2;
			status[i][j] = 1;
		}

	t = 0;	 sumlabel = 1;
	intersection(img,point, t, label, tx, ty);
	line(point, t, label, status, sumlabel);
	draw(img,point, t, label, sumlabel);
	//cvSaveImage("imgdele.jpg", img);

	int t_vb = 0, t1_vb = 0, kt = 0;
	CvPoint *p_dotted;
	p_dotted = new CvPoint[t];

	for (int sum = 1; sum <= sumlabel; sum++)
	{
		t_vb = 0;		t1_vb = 0;		kt = 0;
		for (int k = 0; k < t; k++) 
		{
			i = point[k].y;		j = point[k].x;
			if (label[i][j] == sum) 
			{
				if (1 == label_Neighbor(label, i, j, sum)) 
				{
					p_dotted[t1_vb] = point[k];
					t1_vb = t1_vb + 1;
				}				
					t_vb = t_vb + 1;
			}
		}
		if ((t_vb >= 7) && (t_vb <= 25)&& (t1_vb >= 1))
			if ((CheckLine(p_dotted[0], p_dotted[t1_vb - 1], t_vb, threshold_line_dotted) == 1))
			{
				q.x = p_dotted[0].x;			q.y = p_dotted[0].y;
				q1.x = p_dotted[t1_vb - 1].x;	q1.y = p_dotted[t1_vb - 1].y;							
				orient[q.y][q.x] = cvRound(	cvFastArctan((float) abs(q.y - q1.y),(float) abs(q.x - q1.x)));
				orient[q1.y][q1.x] = orient[q.y][q.x];
				label[q1.y][q1.x] = 0;
				label[q.y][q.x] = 0;				
				c.begin = q;  c.end = q1;				
				A.push_back(c);
			}		
	}
	delete[] p_dotted;

	///////////
	double kc1 = 0.0, kc2 = 0.0;
	CvPoint tmp;
	double hsa, hsb, hsc;
	hsa = 0.0;	hsb = 0.0;	hsc = 0.0;
	if (A.size() > 1) 
	{
		for (int k1 = 0; k1 < A.size() - 1; k1++) 
		{
			c = A.at(k1);			
			q = c.begin; q1 = c.end; 
			if (0 == label[q.y][q.x])
			{
				B.push_back(c);
				tmp.x = q1.x - q.x;		tmp.y = q1.y - q.y;
				hsa = -tmp.y;			hsb = tmp.x;
				hsc = (-1) * hsa * q.x - hsb * q.y;
				for (int k2 = k1 + 1; k2 < A.size(); k2++) 
				{
					d = A.at(k2);						
					q = d.begin; q1 = d.end;
					if (abs(orient[d.begin.y][d.begin.x]- orient[c.begin.y][c.begin.x]) <= 10)						
					{
						kc1 = abs(((hsa) * q1.x + (hsb) * q1.y + hsc)/ sqrt((hsa * hsa) + (hsb * hsb)));
						kc2 = abs(((hsa) * q.x + (hsb) * q.y + hsc)	/ sqrt((hsa * hsa) + (hsb * hsb)));
						if (((int) kc1 <= 10) && ((int) kc2 <= 10)) 
						{
							label[d.begin.y][d.begin.x] = 0;
							d.begin.arw = 0;
							B.push_back(d);
						}
					}					
				}
				if (B.size() > 5)
				{					
					check_dot_line( B, dott_line);
					for (i = 1; i < B.size(); i++) 
					{
						d = B.at(i);
						label[d.begin.y][d.begin.x] = 1;// danh dau diem da xet
					}
				}
				if (B.size() > 0)	B.clear();
			}
		}
	}
	A.clear();
}
/*****************************************************************************/
/**check_dot_line: Loc ra cac duong o gan nhau va coi la day net dut 
 * @param  A
 * [in]: Tap cac duong cung huong
 * @param  dot_line
 * [out]: Tap cac net dut 
 *****************************************************************************/
void check_dot_line( vector<Line> A, vector<Line> &dot_line)
{
	int i, j;
	Line a, b;
	vector<Line> B;	
	A.at(0).begin.arw = 1;  // Muc dich danh dau duong da xet
	a = A.at(0);
	B.push_back(a);	
	int kt1 = 0, kt2 = 0, k, t_label;

	while (kt1 == 0)
	{
		k = 0;
		kt2 = 0;
		t_label = 0;
		for (i = 1; i < A.size(); i++) 
		{
			a = A.at(i);			
			if (0 == a.begin.arw) 
			{
				t_label = t_label + 1;
				break;
			}
		}
		if (t_label == 0)
			kt1 = 1;
		else {
			for (i = 1; i < A.size(); i++) 
			{
				a = A.at(i);				
				if (0 == a.begin.arw)				
				{
					kt2 = 0;
					for (j = 0; j < B.size(); j++) 
					{
						b = B.at(j);
						if ((distance_line(a, b) <= 40)&& (distance_line(a, b) >= 5)) 
						{
							B.push_back(a);
							k = k + 1;							
							A.at(i).begin.arw = 1;
							break;
						}
					}
				}
			}
		}
		if (k == 0)		kt2 = 1;
		if ((kt2 == 1) && (B.size() > 5))
			reset_dotted_line(B, dot_line);
		if (kt2 == 1)	kt1 = 1;
	}
}
/*****************************************************************************/
/**reset_dotted_line: thiet lap lai cac duong net dut
/ xac dinh lai cac duong net dut theo ti le (do dai 1 net /khoang cach giua 2 net = 1/2)
 * @param  A
 * [in]: Tap cac net dut
 * @param  B
 * [out]: Tap cac net dut duoc xac dinh lai
 * return: none
 *****************************************************************************/
void reset_dotted_line(vector<Line> A, vector<Line>&B) 
{
	cvpoint q, q1;
	float length;
	int i, t3 = A.size();
	Line c;
	int minx = 10000, maxx = 0, miny = 10000, maxy = 0, delta = 0;
	/// xac dinh so net dut va xac dinh diem dau, diem cuoi cua day cac net dut 
	for (i = 0; i < t3; i++) 
	{
		c = A.at(i);		
		q = c.begin; q1 = c.end;
		if (minx > min(q.x,q1.x))	minx = min(q.x,q1.x);
		if (maxx < max(q.x,q1.x))	maxx = max(q.x,q1.x);
		if (miny > min(q.y,q1.y))	miny = min(q.y,q1.y);
		if (maxy < max(q.y,q1.y))	maxy = max(q.y,q1.y);		
	}
	// xac dinh do dai duong net dut
	length = sqrt((float) (maxx - minx) * (maxx - minx)
				+ (float) (maxy - miny) * (maxy - miny));
	delta = (int) length / (t3 + (t3 - 1) * 2); // tinh khoang cach giua cac dotted_line.	
	if (delta>=8)
	{
		float goc = cvRound(cvFastArctan((float) maxy - miny, (float) maxx - minx));
		if (goc <= 5)	goc = 0;
		if (goc >= 85)	goc = 90;
		q.y = miny;	q.x = minx;
		if ((goc <= 5) || (goc >= 85))
		{
			q.y = miny;		q.x = minx;
		}
		else 
		{
			c = A.at(0);		
			q = c.begin; q1 = c.end;
			if ((maxx == q.x) || (maxx == q1.x)) 
			{
				q.x = maxx;		q.y = miny;
			}
		}
		double sina = sin(goc * 3.1416 / 180);
		double cosa = cos(goc * 3.1416 / 180);	
		if ((q.x == minx) && (q.y == miny))
			for (i = 0; i < t3; i++) 
			{
				q1.x = q.x + (int) (delta * cosa);
				q1.y = q.y + (int) (delta * sina);			
				c.begin = q; c.end = q1;
				B.push_back(c);			
				q.x = q1.x + (int) (cosa * 2 * delta);
				q.y = q1.y + (int) (sina * 2 * delta);
			}
		if ((q.x == maxx) && (q.y == miny))
			for (i = 0; i < t3; i++) 
			{
				q1.x = q.x - (int) (delta * cosa);
				q1.y = q.y + (int) (delta * sina);			
				c.begin = q; c.end = q1;
				B.push_back(c);			
				q.x = q1.x - (int) (cosa * 2 * delta);
				q.y = q1.y + (int) (sina * 2 * delta);
			}
	}
}
/*****************************************************************************/
/**distance_line: Tinh khoang cach giua 2 duong thang cung huong voi nhau
 * @param  line1, line2
 * [in]: 2 duong can tinh khoang cach
 * return: dis: khoang cach giua 2 duong thang
 *****************************************************************************/
int distance_line(Line line1, Line line2) 
{
	int dis, min = 100;
	float length[4];

	length[0] = sqrt(float(line1.begin.x - line2.begin.x)* float(line1.begin.x - line2.begin.x)
				   + float(line1.begin.y - line2.begin.y)* float(line1.begin.y - line2.begin.y));
	length[1] = sqrt(float(line1.begin.x - line2.end.x)* float(line1.begin.x - line2.end.x)
				   + float(line1.begin.y - line2.end.y)* float(line1.begin.y - line2.end.y));
	length[2] = sqrt(float(line1.end.x - line2.begin.x)* float(line1.end.x - line2.begin.x)
				   + float(line1.end.y - line2.begin.y)* float(line1.end.y - line2.begin.y));
	length[3] = sqrt(float(line1.end.x - line2.end.x)* float(line1.end.x - line2.end.x)
				   + float(line1.end.y - line2.end.y)* float(line1.end.y - line2.end.y));

	for (int i = 0; i < 4; i++) {
		if (min > length[i])
			min = (int) length[i];
	}
	dis = min;
	return dis;
}
/*****************************************************************************/
/**CheckLine: Kiem tra 1 line có là duong thang hay khong
 * @param  begin, end
 * [in]: Toa do diem dau, cuoi cua duong
 * @param  length_line
 * [in]: do dai cua duong
 * @param threshold_check_line
 * [in]: nguong de kiem tra co la duong thang
 * return: check_is_line: 1: là duong thang ; 0: khong la duong thang
 *****************************************************************************/
int CheckLine(CvPoint begin, CvPoint end, int length_line, float threshold_check_line)
{
	int check_is_line = 0;
	float t1 = (float) length_line;
	float t2 = t1 / abs( begin.x - end.x );
	float t3 = t1 / abs( begin.y - end.y );
	if (length_line >= 5)
		if ((t2 < threshold_check_line) || (t3 < threshold_check_line))
			check_is_line = 1;
	return (check_is_line);
}
/*****************************************************************************/
/**AngleOf2Lines: goc giua 2 duong
 * @param  line1, line2
 * [in]: 2 duong can tinh goc
 * return: angle
 *****************************************************************************/
int AngleOf2Lines(Line line1, Line line2) 
{
	CvPoint vector1, vector2;
	vector1.x = line1.begin.x - line1.end.x;
	vector1.y = line1.begin.y - line1.end.y;
	vector2.x = line2.begin.x - line2.end.x;
	vector2.y = line2.begin.y - line2.end.y;

	double lvector1 = sqrt(double(vector1.x * vector1.x + vector1.y * vector1.y));
	double lvector2 = sqrt(double(vector2.x * vector2.x + vector2.y * vector2.y));
	double mul_vector  =   double(vector1.x * vector2.x + vector1.y * vector2.y);
	int angle;
	if (abs(mul_vector) > abs(lvector1 * lvector2)) {
		angle = (int) abs(
				acos((lvector1 * lvector2) / mul_vector) * 180 / CV_PI);
	} else {
		angle = (int) abs(
				acos(mul_vector / (lvector1 * lvector2)) * 180 / CV_PI);
	}
	return angle;
}
