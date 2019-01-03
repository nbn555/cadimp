#include "windows.h"
#include <vector>
#include "TextDetectorAPI.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "LiProcess.h"
#include "shape_detection.h"
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src, src_gray;

	/// Read the image
	src = imread("Test3\\20181219031311074_0014.jpg", 1);

	if (!src.data)
	{
		return -1;
	}
	Mat removedBorderMat;
	removeBorder(src, removedBorderMat);
	//imshow("removed", removedBorderMat);
	//waitKey();
	//vector<Point> intersectionPoints;
	///*
	//	for (size_t i = 0; i < intersectionPoints.size(); i++)
	//	{
	//		circle(removedBorderMat, intersectionPoints[i], 4, Scalar(0, 0, 255), 2);
	//	}
	//	imshow("removedBorderMat", removedBorderMat);
	//	waitKey();*/
	//	vector<Vec3f> outCircles;
	//detectCircle(removedBorderMat,outCircles, intersectionPoints, "");
    char result[MAX_PATH];
    GetCurrentDir(result, sizeof(result));
  //  std::string trainpath = std::string(result);
    //setTrainingDataPath(std::string(result));
    setMinHeightText(5);
	vector<pair<string, RotatedRect>> outText;
	detectText(removedBorderMat, outText);
	return EXIT_SUCCESS;
}