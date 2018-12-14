#include "windows.h"
#include <vector>
#include "TextDetectorAPI.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
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
	src = imread("..\\images\\file 1_8.png", 1);
	if (!src.data)
	{
		return -1;
	}
    char result[MAX_PATH];
    GetCurrentDir(result, sizeof(result));
  //  std::string trainpath = std::string(result);
    setTrainingDataPath(std::string(result));
	vector<pair<string, RotatedRect>> outText;
	detectText(src, outText);
	return EXIT_SUCCESS;
}