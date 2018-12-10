#include"preprocessing.h"
//////////////////////////////////////////////
/*****************************************************************************/
/**RectangularRecognition: Nhan dang hinh chu nhat
/* Nhan dang hinh chu nhat: xet ti le dien tich hinh/ dien tich hcn bao quanh //
 * @param  img
 * [in_out]: Anh 
 * @param  cont
 * [in]: Toa do cac diem anh cua mot hinh.
 * @param circle
 * [out]: du lieu ve hinh chu nhat: toa do diem goc tren trai va goc duoi phai
 * @param k
 * [out]: 1: la hinh chu nhat.   
 * @return: none   
 *****************************************************************************/
void RectangularRecognition(IplImage* img, CvSeq* con, vector<RectangularList> &rec, int &k)
{
	float sumx =0, sumy =0, length =0.0f, max_length = 0, min_length = 1000;
	int cs_max =0;
	float x_center, y_center, hsa, hsb, hsc;
	float dien_tich = 0, area, ti_le;
	int chieu_cao = 0;

	for(;con;con=con->h_next)
	{			
		CvRect feret=cvBoundingRect(con, 0);
		sumx = 0; sumy = 0; max_length =0; min_length =10000;
		 for( int i=0; i<con->total; ++i )			
		  {
			CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, con, i );										
			sumx = sumx + p->x;
			sumy = sumy + p->y;									
		  }		
		x_center = sumx/float(con->total);
		y_center = sumy/float(con->total);
		dien_tich = 0.0f;

		// tinh dien tich cua hinh
		for( int i=0; i<con->total-1; ++i )		
		{  
			CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, con, i );							
			CvPoint* q = CV_GET_SEQ_ELEM( CvPoint, con, i+1 );						  
			length = sqrt(float(p->x - x_center)* float(p->x - x_center) 
						+ float(p->y - y_center)* float(p->y - y_center));
			if(length <  min_length)	min_length = length;
			hsa= -(p->y - q->y);	hsb= p->x - q->x;	hsc = (-1)*hsa*q->x - hsb*q->y;
			chieu_cao = abs(((hsa)*x_center + (hsb)*y_center + hsc)/sqrt((hsa*hsa)+(hsb*hsb)));								  							  
			dien_tich = dien_tich + chieu_cao *1.0;	
		}																	
		dien_tich = (0.5)*dien_tich;								
		// end//
		area = fabs(cvContourArea(con,CV_WHOLE_SEQ));// tinh dien tich hcn bao quanh 								
		ti_le = dien_tich/area;	
		if ((ti_le > 0.92)&&((int)min_length >5)) // dieu kien xet la HCN
		{
			RectangularList rectangular;
			rectangular.xmin = feret.x;
			rectangular.ymin = feret.y;
			rectangular.ymax = feret.y + feret.height;
			rectangular.xmax = feret.x + feret.width;
			rec.push_back(rectangular);
			k = 1;
		}
		if (1 == k) break;
	}	
}

/*****************************************************************************/
/**CircleRecognition: Nhan dang hinh tron
// Nhan dang hinh tron: xet ti le so luong diem co khoang cach toi tam gan bang ban kinh//
// voi tong so luong diem lon hon 0.8 thi nhan dang la hinh tron//
 * @param  img
 * [in_out]: Anh 
 * @param  cont
 * [in]: Toa do cac diem anh cua mot hinh.
 * @param circle
 * [out]: du lieu ve hinh tron: tam, ban kinh
 * @param k
 * [out]: 1: la hinh tron.   
 * @return: none   
 *****************************************************************************/
void CircleRecognition(IplImage* img, CvSeq* cont, vector<CircleList> &circle,int &k)
		{
			for(;cont;cont=cont->h_next)
			{
				CvRect feret=cvBoundingRect(cont, 0);
				double area = fabs(cvContourArea(cont,CV_WHOLE_SEQ));
				double feret_ratio=((feret.width<feret.height)?(double)feret.width/(double)feret.height:
						(double)feret.height/(double)feret.width);
				int thres = (img->width > img->height ? img->width:img->height)/2;
				int flag = (cont->flags & CV_SEQ_FLAG_HOLE);
				// xac dinh tam, ban kinh, so luong diem co khoang cach toi tam gan bang ban kinh.
				if((area > thres)&&(flag == 0))
				{
					int cx, cy;
					cx = feret.x + feret.width/2;
					cy = feret.y + feret.height/2;
					int check = 0;
					if((feret_ratio >= 0.75)&& (feret.width > 10))
					{							
						for( int i=0; i < cont->total; ++i ) {
							CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, cont, i );
							double length = sqrt(double((p->x - cx)*(p->x - cx) + (p->y - cy)*(p->y - cy)));
							if(abs(length - feret.width/2) < double(feret.width)/10)							
							{													
								check++;
							}
						}
					}			
				// xet ti le nhan dang hinh tron
					if(double(check)/double(cont->total) > 0.75)
					{					
						CircleList cir;
						cir.x_center = cx;
						cir.y_center = cy;
						cir.radius = feret.width/2;
						circle.push_back(cir);
						k =1;
					}
				}				
				if (1 == k) 
				{	
					break;
				}
			}
		}

/*****************************************************************************/
/**EllipseRecognition: Nhan dang elip: goc giua truc lon, truc nho ~ 90//
 * @param  img
 * [in_out]: Anh 
 * @param  cont
 * [in]: Toa do cac diem anh cua mot hinh.
 * @param ellip
 * [out]: du lieu ve elip: tam, truc lon, truc nho, huong elip
 * @param k
 * [out]: 1 la elip.   
 * @return: none   
 *****************************************************************************/
void EllipseRecognition(IplImage* img, CvSeq* cont, vector<Ellip> &ellip, int &k) 
{
	for (; cont; cont = cont->h_next) 
	{
		CvRect feret = cvBoundingRect(cont, 0);
		double area = fabs(cvContourArea(cont, CV_WHOLE_SEQ));
		double feret_ratio =
				((feret.width < feret.height) ? (double)feret.width/(double)feret.height:
				(double)feret.height/(double)feret.width);
		int thres = (	img->width > img->height ?
						img->width : img->height)/ 2;
		int flag = (cont->flags & CV_SEQ_FLAG_HOLE);
		// xac dinh tam, truc lon, truc nho
		if ((area > thres) && (flag == 0)) 
		{
			double sumx = 0;
			double sumy = 0;
			for (int i = 0; i < cont->total; ++i) 
			{
				CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, cont, i);
				sumx = sumx + p->x;
				sumy = sumy + p->y;
			}
			double x_center = sumx / double(cont->total);
			double y_center = sumy / double(cont->total);

			double max_length = 0;
			double min_length = 10000;
			double length = 0;
			int maxx, maxy, minx, miny;
			for (int i = 0; i < cont->total; ++i)
			{
				CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, cont, i);
				length = sqrt(	double(p->x - x_center)	* double(p->x - x_center)
							  + double(p->y - y_center) * double(p->y - y_center));
				if (length > max_length) 
				{
					max_length = length;
					maxx = p->x;
					maxy = p->y;
				}
				if (length < min_length) 
				{
					min_length = length;
					minx = p->x;
					miny = p->y;
				}
			}
			double angle = acos(
					fabs((maxx - x_center) * (minx - x_center)
						+(maxy - y_center) * (miny - y_center))
						/(min_length * max_length)) * 180.0/ CV_PI;

			if (fabs(angle - 90.0) <= 10) // goc giua truc lon truc nho thoa man nguong
			{				
				double sum = 0;
				for (int i = 0; i < cont->total; ++i) 
				{
					CvPoint* p = CV_GET_SEQ_ELEM(CvPoint, cont, i);
					length = sqrt(
							double(p->x - x_center)* double(p->x - x_center)
							+ double(p->y - y_center)* double(p->y - y_center));
					if (length < min_length) 
					{
						sum++;
					}
				}
				if (((sum / cont->total) < 0.1)
					&& ((min_length / max_length) < 0.9) && (min_length>= 7)) // truc nho < truc lon
				{
					int x = (int) x_center;
					int y = (int) y_center;

					CvSize size = cvSize((int) feret.width / 2,	(int) feret.height / 2);
					CvSize size1 = cvSize((int) max_length,	(int) min_length);
					CvPoint pt = cvPoint(x, y);
					int goc = cvRound(cvFastArctan((float) (y - maxy), (float)(maxx - x)));
					Ellip el;
					el.x_center = x;
					el.y_center = y;
					el.angle = goc;
					if ((goc <= 10) || (goc >= 350)
							|| ((goc >= 170) && (goc <= 190))) 
					{
						goc = 0;
						el.angle = goc;	
						el.max_radius = feret.width / 2;
						el.min_radius = feret.height / 2;
					} else {
						el.max_radius = (int) max_length;
						el.min_radius = (int) min_length;
					}
					ellip.push_back(el);
					k = 1;
				}
			}
		}
		if (1 == k)
		{			
			break;
		}
	}

		}
		

/////////////////////

