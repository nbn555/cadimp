#include "windows.h"
#include <vector>
#include "TextDetectorAPI.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "LiProcess.h"
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
	src = imread("300 Dpi\\2\\20181219031311074_0001.jpg", 1);
	if (!src.data)
	{
		return -1;
	}
	Mat removedBorderMat;
	removeBorder(src, removedBorderMat);
	imshow("removed", removedBorderMat);
	waitKey();
 //   char result[MAX_PATH];
 //   GetCurrentDir(result, sizeof(result));
 // //  std::string trainpath = std::string(result);
 //   setTrainingDataPath(std::string(result));
 //   setMinHeightText(5);
	//vector<pair<string, RotatedRect>> outText;
	//detectText(src, outText);
	return EXIT_SUCCESS;
}