#ifndef IMGCATCH_LIPROCESS_H
#define IMGCATCH_LIPROCESS_H

#include"preprocessing.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
using namespace cv;

//#define  SHOW_IMAGE_TO_DEBUG
///// track_point//////////
void detectCircle(Mat &src, vector<Vec3f> &outCircles);
void CreateIntMatrix(int **&matrix, CvSize size);
void ReleaseIntMatrix(int **&matrix, int size);
void intersection(IplImage *img,CvPoint *point, int &t, int **label,int **tx,int **ty); 
void connect_point_3(CvPoint *point, int t, int **label, int **tx,	int **ty);
void connect_point_34(CvPoint *point, int t, int **label, int **tx,	int **ty);
void point3_point3(int **label, int y1, int x1, int y2, int x2);
void line(CvPoint *point, int t, int **label, int **status, int &sumlabel); 
void check_point(CvPoint *point, int t, int **label, int **status, int &sumlabel); 
void connect_2line(CvPoint *point, int t, int ** label, int **status, int y1,int x1, int y2, int x2);

void orientline(CvPoint *point, int t,int **status, int **label, int **orient,int param1);
int orient_Neighbor(int **label, int y, int x, int label_xy);
void find_orientline(int **label,int **orient, int y, int x,int param1);

void link (CvPoint *point,int t, int **orient, int **label, int **status, int i,int j,
			int y, int x, int min,int param2,int &kt);		

void link3(CvPoint *point,int t, int **orient, int **label, int **status,int **tx, int **ty,
		   int y1,int x1,int y2, int x2, int d,int param2);
void find_min(CvPoint *point, int t, int **orient, int **status, int i, int j,int y, int x, int &min);

void connect_line(CvPoint *point, int t, int **label,int **status,int **orient,	int **tx, int **ty, int param2,int &sumlabel);
void connect_gd(CvPoint *point, int &t,int **tx, int **ty,int **orient, int **label, int **status,
				int sumlabel,int param2);       
void connect_31(CvPoint *point, int t,int **tx, int **ty,int **orient, int **label,	int **xet, int param2);
void connect_4(CvPoint *point, int t, int **label, int **status, int **orient, int **tx, int **ty, int param2);
void connect_3(CvPoint *point, int t,int **tx, int **ty,int **orient, int **label, int **status, int param2);       
void connect_other(CvPoint *point, int t, int **label, int **status,int **orient,int param2);

int label_Neighbor(int **label, int y, int x, int d);
int Neighbor_3(int **label, int y, int x);

void connect(IplImage *img, CvPoint *point, int t, int **label, int **status,Line &c, int &check);

//void connect_point(IplImage *img, CvPoint *point, int t, int t_label,int **status);
void connect_point(IplImage *img, CvPoint *point, int t, int t_label,int **status, CvSeq* lcontour);
void draw(IplImage *img,CvPoint *point, int t, int **label, int &sumlabel);
void track_point(IplImage *src,int param1, int param2, vector<RectangularList> &rec, 
			vector<CircleList> &circle,	vector<Ellip> &ellip, vector<Line> &A, vector<Line> &dott_line,
			vector<RectangularList> &rec_text);
void draw_out(IplImage *src,vector<RectangularList> &rec, 
			vector<CircleList> &circle,	vector<Ellip> &ellip, vector<Line> &A, vector<Line> &dott_line,
			vector<RectangularList> &rec_text);
void line_arrow(int ** label, int x, int y, int Line_Arr[10],int &sumlabel_arr);
void arrow( vector<Line> &A, CvPoint *point, int t,
		int ** label, int **orient, int sumlabel);
void solid_line(IplImage *img, CvPoint *point, int t, Line &line_solid, vector<Line> &A);
void dotted_line(IplImage *img, CvPoint *point, int t, int **label,
		int ** status, int **orient, int **tx, int **ty,
		int &sumlabel, vector<Line> &A);
void check_dot_line(vector<Line> A, vector<Line>&dot_line);
int distance_line(Line line1, Line line2);
void reset_dotted_line(vector<Line> A, vector<Line>&B);
int CheckLine(CvPoint begin, CvPoint end, int length_line, float threshold_check_line);
void Check_arrow(cvpoint &p, int orient_line1, Line line1, CvPoint *point,
		int t, int **label, int Line_Arr[10], int sumlabel_arr);
int AngleOf2Lines(Line line1, Line line2);
void rect_text(CvPoint *point, int t,int **label,int sumlabel,vector<RectangularList> &rec_text);
void join2hcn( RectangularList &hcn1, RectangularList &hcn2, int &kt);
void connect_solid_line(vector<Line> A,vector<Line> &LineA);
///////////end_track_point//////////


// Nguong ket noi 2 diem giao nhau
#define pixel_connect_intersection		7//10
// Nguong xac dinh cac diem thuoc duong ngan can xoa
#define threshold_delete_point			10
// Nguong do dai cua 1 duong thang
#define threshold_pixel_line			35

#endif