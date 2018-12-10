#ifndef IMGCATCH_PREPROCESSING_H
#define IMGCATCH_PREPROCESSING_H

#include<stdio.h>
#include<conio.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "math.h"
using namespace std;
using namespace cv;


// Cau truc HCN
typedef struct {
	int xmin, ymin, xmax, ymax;		// toa do diem tren trai, duoi phai
} RectangularList;
// Cau truc hinh tron
typedef struct {
	int x_center, y_center;			// tam
	double radius;					// ban kinh
} CircleList;
// Cau truc Elip
typedef struct {
	int x_center, y_center;			// tam
	double max_radius, min_radius;	// do dai truc lon, truc nho
	int angle;						// huong elip	
} Ellip;

//////////////////////////////////////////////////////////////
// Cau truc diem
typedef struct {
	int x;							// hoanh do
	int y;							// tung do
	int arw;						// 1: co mui ten; 0: khong co mui ten
} cvpoint;
// Cau truc duong thang
typedef struct {
	cvpoint begin;					// diem dau 
	cvpoint end;					// diem cuoi
} Line;

// dinh nghia gia tri diem anh
#define data(img, x, y) ((uchar*)(img)->imageData)[(y) * (img)->widthStep + (x) * (img)->nChannels]

IplImage *nhiphan(IplImage *img_input);
void Stentiford(IplImage *img);


///////////nhan dang hinh////////
void RectangularRecognition(IplImage* img, CvSeq* cont, vector<RectangularList> &rec, int &k);
void CircleRecognition(IplImage* img, CvSeq* cont, vector<CircleList> &circle, int &k);
void EllipseRecognition(IplImage* img, CvSeq* cont, vector<Ellip> &ellip, int &k);

#endif