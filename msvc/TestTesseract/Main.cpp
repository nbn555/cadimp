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
	//detectTextOfFolder("Test3.3");
	/// Read the image
	//src = imread("G:/Nghi/ImgCatch/300 Dpi/2/20181219031311074_0005.jpg", 1);
	src = imread("Test3/t (4).jpg");
	if (!src.data)
	{
		return -1;
	}
	vector<Point> intersectionPoints;
	vector<Vec3f> outCircles;
	Mat removedBorderMat;
	removeBorder(src, removedBorderMat);
	/*intersectionPoints.push_back(Point(2044, 1491));
	intersectionPoints.push_back(Point(2042, 1491));
	intersectionPoints.push_back(Point(2043, 682 ));
	intersectionPoints.push_back(Point(2044, 537 ));
	intersectionPoints.push_back(Point(1866, 1466));
	intersectionPoints.push_back(Point(1199, 1491));
	intersectionPoints.push_back(Point(1638, 1491));
	intersectionPoints.push_back(Point(1898, 1491));
	intersectionPoints.push_back(Point(2221, 1491));
	intersectionPoints.push_back(Point(2060, 1491));
	intersectionPoints.push_back(Point(942, 1786 ));
	intersectionPoints.push_back(Point(1898, 1784));
	intersectionPoints.push_back(Point(942, 1661 ));
	intersectionPoints.push_back(Point(1898, 1659));
	intersectionPoints.push_back(Point(1199, 1660));
	intersectionPoints.push_back(Point(1638, 1659));
	intersectionPoints.push_back(Point(1216, 1660));
	intersectionPoints.push_back(Point(1896, 683 ));
	intersectionPoints.push_back(Point(1896, 1255));
	intersectionPoints.push_back(Point(2218, 537 ));
	intersectionPoints.push_back(Point(2058, 537 ));
	intersectionPoints.push_back(Point(2058, 682 ));
	intersectionPoints.push_back(Point(2060, 1254));
	intersectionPoints.push_back(Point(964, 575	 ));
	intersectionPoints.push_back(Point(1199, 1311));
	intersectionPoints.push_back(Point(1199, 1256));
	intersectionPoints.push_back(Point(1199, 1199));
	intersectionPoints.push_back(Point(1198, 739 ));
	intersectionPoints.push_back(Point(1198, 685 ));
	intersectionPoints.push_back(Point(1142, 685 ));
	intersectionPoints.push_back(Point(1140, 685 ));
	intersectionPoints.push_back(Point(1254, 685 ));
	intersectionPoints.push_back(Point(1580, 684 ));
	intersectionPoints.push_back(Point(1637, 684 ));
	intersectionPoints.push_back(Point(950, 563	 ));
	intersectionPoints.push_back(Point(1639, 1311));
	intersectionPoints.push_back(Point(1638, 1198));
	intersectionPoints.push_back(Point(1637, 627 ));
	intersectionPoints.push_back(Point(1868, 1467));
	intersectionPoints.push_back(Point(1866, 1465));
	intersectionPoints.push_back(Point(1142, 1256));
	intersectionPoints.push_back(Point(1582, 1255));
	intersectionPoints.push_back(Point(1696, 1255));
	intersectionPoints.push_back(Point(1696, 1255));
	intersectionPoints.push_back(Point(1162, 637 ));
	intersectionPoints.push_back(Point(1199, 1199));
	getIntersections(removedBorderMat, intersectionPoints, 1);*/
	//detectCircle(removedBorderMat,outCircles, intersectionPoints, "");
	std::vector<std::pair<int, int>> circleInfors;
	circleInfors.push_back(std::pair<int, int>(50, 1));
	//circleInfors.push_back(std::pair<int, int>(8, 1));
	detectCircle(removedBorderMat, intersectionPoints, circleInfors, outCircles);
 //   char result[MAX_PATH];
 //   GetCurrentDir(result, sizeof(result));
 // //  std::string trainpath = std::string(result);
 //   //settrainingdatapath(std::string(result));
 //   setMinHeightText(5);
	/*vector<pair<string, RotatedRect>> outtext;
	detectText(removedBorderMat, outtext);*/
	return EXIT_SUCCESS;
}