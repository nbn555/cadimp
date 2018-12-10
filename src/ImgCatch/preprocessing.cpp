#include"preprocessing.h"

/*****************************************************************************/
/**dem: dem so diem trang trong lan can 3x3 cua (x,y)
 * @param  A[]
 * [in]: Mang gia tri diem anh cac lang gieng cua (x,y)
 * @return: tong: so luong lang gieng cua diem anh
 *****************************************************************************/
int dem(int A[]) {
	int tong = 0;
	for (int i = 1; i <= 8; i++)
		if (A[i] != 0)
			++tong;
	return tong; // tra ve so lang gieng voi (x,y)
}

/*****************************************************************************/
/**langgieng: xac dinh gia tri cac diem anh lan can 3x3
 * @param  img
 * [in]: Anh
 * @param  x,y
 * [in]: Toa do (x,y)
 * @param  A[]
 * [out]: Mang gia tri diem anh cac lang gieng cua (x,y)
 * @return: none
 *****************************************************************************/
void langgieng(IplImage *img, int x, int y, int A[]) {
	A[1] = data(img, x + 1, y);
	A[2] = data(img, x + 1, y + 1);
	A[3] = data(img, x, y + 1);
	A[4] = data(img, x - 1, y + 1);
	A[5] = data(img, x - 1, y);
	A[6] = data(img, x - 1, y - 1);
	A[7] = data(img, x, y - 1);
	A[8] = data(img, x + 1, y - 1);
}



/*****************************************************************************/
/**ketnoi: tinh so ket noi xung quanh diem (x,y) trong lan can 3x3
 * @param  A[]
 * [in]: Mang gia tri diem anh cac lang gieng cua (x,y)
 * @return: kq: so ket noi
 *****************************************************************************/
int ketnoi(int A[]) {
	int kq = 0;
	int a[9];// Mang luu tru (gom 2 gia tri 0 va 1)cac diem vung 3x3.
	for (int k = 1; k <= 8; k++) {
		if (A[k] == 255)
			a[k] = 1;
		else
			a[k] = 0;
	}
	// so ket noi tinh theo cong thuc: kq+ = a[i] - a[i].a[i+1].a[i+2]
	kq += a[1] - a[1] * a[2] * a[3]; 
	kq += a[3] - a[3] * a[4] * a[5];
	kq += a[5] - a[5] * a[6] * a[7];
	kq += a[7] - a[7] * a[8] * a[1];
	return kq; // tra ve so ket noi
}


/*****************************************************************************/
/**before_Sentiford: xu li truoc thinning
// diem den nam giua khoi dac duoc fill thanh diem trang.
// fill diem anh 
 * @param  img
 * [in_out]: Anh 
 *****************************************************************************/
void before_Sentiford(IplImage *img_bin)
{
	int sum =0,i,j;	
	uchar* data= (uchar*)img_bin->imageData;	
	for (i = 1; i< img_bin->height-1; i++)
		for (j = 1; j< img_bin->width-1; j++)
		{		   
           if (255 == data[i*img_bin->widthStep+j])
      		{
				sum = 0;
				for (int h = - 1; h<=1; h++)
					for (int w = - 1; w<=1; w++)
						if (255 == data[(i+h)*img_bin->widthStep+(j+w)]) 
							sum = sum + 1;					
				if ((sum <=5)&&(sum>0))// 5 vi tinh ca (i,j)
				{					
					if (255 == data[(i-1)*img_bin->widthStep+j-1])
						if ((0 == data[(i)*img_bin->widthStep+j-1])&& (0 == data[(i-1)*img_bin->widthStep+j]))
						{
							img_bin->imageData[i*img_bin->widthStep+j-1] = (uchar)255 ;
							img_bin->imageData[(i-1)*img_bin->widthStep+j] = (uchar)255 ;
						}

					if (255 == data[(i-1)*img_bin->widthStep+j+1])
						if ((0 == data[(i)*img_bin->widthStep+j+1])&& (0 == data[(i-1)*img_bin->widthStep+j]))
						{
							img_bin->imageData[i*img_bin->widthStep+j+1] = (uchar)255 ;
							img_bin->imageData[(i-1)*img_bin->widthStep+j+1] = (uchar)255 ;
						}

					if (255 == data[(i+1)*img_bin->widthStep+ j-1])
						if ((0 == data[(i+1)*img_bin->widthStep+j])&& (0 == data[i*img_bin->widthStep+j-1]))
						{
							img_bin->imageData[(i+1)*img_bin->widthStep+j] = (uchar)255 ;
							img_bin->imageData[i*img_bin->widthStep+j-1] = (uchar)255 ;
						}

					if (255 == data[(i+1)*img_bin->widthStep+ j+1])
						if ((0 == data[(i+1)*img_bin->widthStep+j])&& (0 == data[i*img_bin->widthStep+j+1]))
						{
							img_bin->imageData[(i+1)*img_bin->widthStep+j] = (uchar)255 ;
							img_bin->imageData[i*img_bin->widthStep+j+1] = (uchar)255 ;
						}
				}			
			}		   
		}
		/// xet trong lan can 3x3, co 1 diem den thi fill thanh diem trang
		// khi thinning khong tao ra lo trong
		for (i = 2; i< img_bin->height-2; i++)
		for (j = 2; j< img_bin->width -2; j++)
		{		   
           if (0 == data[i*img_bin->widthStep+j])
      		{							
				{
					sum = 0;
				 for (int h = i-1; h<=i+1; h++)
					for (int w = j-1; w<=j+1; w++)
						if (255 == data[h*img_bin->widthStep+w]) 
							sum = sum + 1;	
					if (sum>=8)
						img_bin->imageData[i*img_bin->widthStep+j] = (uchar)255;
				}
				
			}		  		 
		}						
}


/*****************************************************************************/
/**Stentiford: Làm mong duong bang cach xoa cac diem bien
// diem bien duoc xoa khi thoa man co so lang gieng la diem anh >1, so ket noi =1.
 * @param img
 * [in_out]: Anh thinninh 
 * @return: none   
 *****************************************************************************/
void Stentiford(IplImage *img)
{
	/// bat dau thuat toan Sentiford
	int tieptuc = 1;	
	int delpixel;
	int x, y;
	IplImage *copy = cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
	cvCopy(img, copy);
	int A[9];
	// lam mong tu phia duoi len 
	while(tieptuc == 1)
	{		
		delpixel = 0;
		for( y = 1; y < img->height - 1; y++)
			for( x = 1; x < img->width - 1; x++)
			{				
				if(data(img, x, y) == 255 && data(img, x, y + 1) == 0 && data(img, x, y - 1) == 255)
				{
					langgieng(img, x, y, A);
					if(dem(A) > 1 && ketnoi(A) == 1)
					{
						data(copy, x, y) = (uchar)0;						
						delpixel = 1;
					}
					else data(img, x, y) = (uchar)200;
				}
			}
		cvCopy(copy, img);

		//////////////////
		// lam mong tu phia trai sang 
		for( y = 1; y < img->height - 1; y++)
			for( x = 1; x < img->width - 1; x++)
			{				
				if(data(img, x, y) == 255 && data(img, x - 1, y) == 0 && data(img, x + 1, y) == 255)
				{
					langgieng(img, x, y, A);
					if(dem(A) > 1 && ketnoi(A) == 1)
					{
						data(copy, x, y) = 0;											
						delpixel = 1;
					}
					else data(img, x, y) = (uchar)200;
				}

			}
		cvCopy(copy, img);
		
		////////////////
		// lam mong tu phia tren xuong 
		for( y = 1; y < img->height - 1; y++)
			for( x = 1; x < img->width - 1; x++)
			{				
				if(data(img, x, y) == 255 && data(img, x, y + 1) == 255 && data(img, x, y - 1) == 0)
				{
					langgieng(img, x, y, A);
					if(dem(A) > 1 && ketnoi(A) == 1)
					{
						data(copy, x, y) = 0;						
						delpixel = 1;
					}
					else data(img, x, y) = (uchar)200;
				}

			}
		cvCopy(copy, img);
		////////////////////
		// lam mong tu phia phai sang 
		for( y = 1; y < img->height - 1; y++)
			for( x = 1; x < img->width - 1; x++)
			{				
				if(data(img, x, y) == 255 && data(img, x - 1, y) == 255 && data(img, x + 1, y) == 0)
				{
					langgieng(img, x, y, A);
					if(dem(A) > 1 && ketnoi(A) == 1)
					{
						data(copy, x, y) = 0;						
						delpixel = 1;
					}
					else data(img, x, y) = (uchar)200;
				}

			}
		cvCopy(copy, img);
		if((delpixel == 1)) tieptuc = 1;
		else tieptuc = 0;
	}
	/// end_Sentiford

	/// loai bo cac diem nhieu
	int sum;
	for (int i = 3; i< img->height-3; i++)
		for (int j = 3; j< img->width -3; j++)
		{		   
           if (255 == data(img,j,i))
      		{				
				sum = 0;
				for (int h = i-3; h <= i+3; h++)
					for (int w = j-3; w <= j+3; w++)
						if (255 == data(img,w,h))
							sum = sum + 1;					
				if ((sum <= 3)&&(sum > 0))					
					data(img,j,i) = (uchar) 0;
			}		  
		}	
}


////////////////////////////////////////
// 35: so pixel tham gia vao tinh toan nguong cua diem anh.
/*****************************************************************************/
/**IplImage *nhiphan: chuyen anh gray sang anh binary
 * @param  img
 * [in_out]: in: Anh gray; out:anh binary
 *****************************************************************************/
IplImage *nhiphan(IplImage *img_input)
{
	IplImage *img_bin = cvCreateImage( cvGetSize( img_input), 8, 1);		
	cvSmooth( img_input, img_bin, CV_MEDIAN, 3, 3 );	
	cvAdaptiveThreshold(img_bin, img_bin, 255, CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,45,7); //5			   	
	before_Sentiford(img_bin);	
    return img_bin;
}