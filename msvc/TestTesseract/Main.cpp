#include <vector>
#include "TextDetectorAPI.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src, src_gray;

	/// Read the image
	src = imread("..\\images\\sd1.png", 1);
	if (!src.data)
	{
		return -1;
	}
	
	vector<pair<string, RotatedRect>> outText;
	detectText(src, outText);
	return EXIT_SUCCESS;
}