/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
//#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <algorithm>
#include <iterator>

namespace cv
{

// Classical Hough Transform
struct LinePolar
{
    float rho;
    float angle;
};


struct hough_cmp_gt
{
    hough_cmp_gt(const Point* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const
    {
        return aux[l1].y > aux[l2].y || (aux[l1].y == aux[l2].y && l1 < l2);
    }
    const Point* aux;
};

static void
createTrigTable( int numangle, double min_theta, double theta_step,
                 float irho, float *tabSin, float *tabCos )
{
    float ang = static_cast<float>(min_theta);
    for(int n = 0; n < numangle; ang += (float)theta_step, n++ )
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }
}

static void
findLocalMaximums( int numrho, int numangle, int threshold,
                   const int *accum, std::vector<int>& sort_buf )
{
    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ )
        {
            int base = (n+1) * (numrho+2) + r+1;
            if( accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
                sort_buf.push_back(base);
        }
}
// Multi-Scale variant of Classical Hough Transform

struct hough_index
{
    hough_index() : value(0), rho(0.f), theta(0.f) {}
    hough_index(int _val, float _rho, float _theta)
    : value(_val), rho(_rho), theta(_theta) {}

    int value;
    float rho, theta;
};



/****************************************************************************************\
*                                     Circle Detection                                   *
\****************************************************************************************/

struct EstimatedCircle
{
    EstimatedCircle(Vec3f _c, int _accum) :
        c(_c), accum(_accum) {}
    Vec3f c;
    int accum;
};

static bool cmpAccum(const EstimatedCircle& left, const EstimatedCircle& right)
{
    // Compare everything so the order is completely deterministic
    // Larger accum first
    if (left.accum > right.accum)
        return true;
    else if (left.accum < right.accum)
        return false;
    // Larger radius first
    else if (left.c[2] > right.c[2])
        return true;
    else if (left.c[2] < right.c[2])
        return false;
    // Smaller X
    else if (left.c[0] < right.c[0])
        return true;
    else if (left.c[0] > right.c[0])
        return false;
    // Smaller Y
    else if (left.c[1] < right.c[1])
        return true;
    else if (left.c[1] > right.c[1])
        return false;
    // Identical - neither object is less than the other
    else
        return false;
}

inline Vec3f GetCircle(const EstimatedCircle& est)
{
    return est.c;
}

class NZPointList : public std::vector<Point>
{
private:
    NZPointList(const NZPointList& other); // non-copyable

public:
    NZPointList(int reserveSize = 256)
    {
        reserve(reserveSize);
    }
};

class NZPointSet
{
private:
    NZPointSet(const NZPointSet& other); // non-copyable

public:
    Mat_<uchar> positions;

    NZPointSet(int rows, int cols) :
        positions(rows, cols, (uchar)0)
    {
    }

    void insert(const Point& pt)
    {
        positions(pt) = 1;
    }

    void insert(const NZPointSet& from)
    {
        cv::bitwise_or(from.positions, positions, positions);
    }

    void toList(NZPointList& list) const
    {
        for (int y = 0; y < positions.rows; y++)
        {
            const uchar *ptr = positions.ptr<uchar>(y, 0);
            for (int x = 0; x < positions.cols; x++)
            {
                if (ptr[x])
                {
                    list.push_back(Point(x, y));
                }
            }
        }
    }
};

class HoughCirclesAccumInvoker : public ParallelLoopBody
{
public:
    HoughCirclesAccumInvoker(const Mat &_edges, const Mat &_dx, const Mat &_dy, int _minRadius, int _maxRadius, float _idp,
                             std::vector<Mat>& _accumVec, NZPointSet& _nz, Mutex& _mtx) :
        edges(_edges), dx(_dx), dy(_dy), minRadius(_minRadius), maxRadius(_maxRadius), idp(_idp),
        accumVec(_accumVec), nz(_nz), mutex(_mtx)
    {
        acols = cvCeil(edges.cols * idp), arows = cvCeil(edges.rows * idp);
        astep = acols + 2;
    }

    ~HoughCirclesAccumInvoker() { }

    void operator()(const Range &boundaries) const
    {
        Mat accumLocal = Mat(arows + 2, acols + 2, CV_32SC1, Scalar::all(0));
        int *adataLocal = accumLocal.ptr<int>();
        NZPointSet nzLocal(nz.positions.rows, nz.positions.cols);
        int startRow = boundaries.start;
        int endRow = boundaries.end;
        int numCols = edges.cols;

        if(edges.isContinuous() && dx.isContinuous() && dy.isContinuous())
        {
            numCols *= (boundaries.end - boundaries.start);
            endRow = boundaries.start + 1;
        }

        // Accumulate circle evidence for each edge pixel
        for(int y = startRow; y < endRow; ++y )
        {
            const uchar* edgeData = edges.ptr<const uchar>(y);
            const short* dxData = dx.ptr<const short>(y);
            const short* dyData = dy.ptr<const short>(y);
            int x = 0;

            for(; x < numCols; ++x )
            {
#if CV_SIMD128
                {
                    v_uint8x16 v_zero = v_setzero_u8();

                    for(; x <= numCols - 32; x += 32) {
                        v_uint8x16 v_edge1 = v_load(edgeData + x);
                        v_uint8x16 v_edge2 = v_load(edgeData + x + 16);

                        v_uint8x16 v_cmp1 = (v_edge1 == v_zero);
                        v_uint8x16 v_cmp2 = (v_edge2 == v_zero);

                        unsigned int mask1 = v_signmask(v_cmp1);
                        unsigned int mask2 = v_signmask(v_cmp2);

                        mask1 ^= 0x0000ffff;
                        mask2 ^= 0x0000ffff;

                        if(mask1)
                        {
                            x += trailingZeros32(mask1);
                            goto _next_step;
                        }

                        if(mask2)
                        {
                            x += trailingZeros32(mask2 << 16);
                            goto _next_step;
                        }
                    }
                }
#endif
                for(; x < numCols && !edgeData[x]; ++x)
                    ;

                if(x == numCols)
                    continue;
#if CV_SIMD128
_next_step:
#endif
                float vx, vy;
                int sx, sy, x0, y0, x1, y1;

                vx = dxData[x];
                vy = dyData[x];

                if(vx == 0 && vy == 0)
                    continue;

                float mag = std::sqrt(vx*vx+vy*vy);

                if(mag < 1.0f)
                    continue;

                Point pt = Point(x % edges.cols, y + x / edges.cols);
                nzLocal.insert(pt);

                sx = cvRound((vx * idp) * 1024 / mag);
                sy = cvRound((vy * idp) * 1024 / mag);

                x0 = cvRound((pt.x * idp) * 1024);
                y0 = cvRound((pt.y * idp) * 1024);

                // Step from min_radius to max_radius in both directions of the gradient
                for(int k1 = 0; k1 < 2; k1++ )
                {
                    x1 = x0 + minRadius * sx;
                    y1 = y0 + minRadius * sy;

                    for(int r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, r++ )
                    {
                        int x2 = x1 >> 10, y2 = y1 >> 10;
                        if( (unsigned)x2 >= (unsigned)acols ||
                            (unsigned)y2 >= (unsigned)arows )
                            break;

                        adataLocal[y2*astep + x2]++;
                    }

                    sx = -sx; sy = -sy;
                }
            }
        }

        { // TODO Try using TLSContainers
            AutoLock lock(mutex);
            accumVec.push_back(accumLocal);
            nz.insert(nzLocal);
        }
    }

private:
    const Mat &edges, &dx, &dy;
    int minRadius, maxRadius;
    float idp;
    std::vector<Mat>& accumVec;
    NZPointSet& nz;

    int acols, arows, astep;

    Mutex& mutex;
};

class HoughCirclesFindCentersInvoker : public ParallelLoopBody
{
public:
    HoughCirclesFindCentersInvoker(const Mat &_accum, std::vector<Point> &_centers, int _accThreshold, Mutex& _mutex) :
        accum(_accum), centers(_centers), accThreshold(_accThreshold), _lock(_mutex)
    {
        acols = accum.cols;
        arows = accum.rows;
        adata = accum.ptr<int>();
    }

    ~HoughCirclesFindCentersInvoker() {}

    void operator()(const Range &boundaries) const
    {
        //int startRow = boundaries.start;
        //int endRow = boundaries.end;
        //std::vector<int> centersLocal;
        //bool singleThread = (boundaries == Range(1, accum.rows - 1));

        //startRow = max(1, startRow);
        //endRow = min(arows - 1, endRow);

        ////Find possible circle centers
        //for(int y = startRow; y < endRow; ++y )
        //{
        //    int x = 1;
        //    int base = y * acols + x;

        //    for(; x < acols - 1; ++x, ++base )
        //    {
        //        if( adata[base] > accThreshold &&
        //            adata[base] > adata[base-1] && adata[base] >= adata[base+1] &&
        //            adata[base] > adata[base-acols] && adata[base] >= adata[base+acols] )
        //            centersLocal.push_back(base);
        //    }
        //}

        //if(!centersLocal.empty())
        //{
        //    if(singleThread)
        //        centers = centersLocal;
        //    else
        //    {
        //        AutoLock alock(_lock);
        //        centers.insert(centers.end(), centersLocal.begin(), centersLocal.end());
        //    }
        //}
    }

private:
    const Mat &accum;
    std::vector<Point> centers;
    int accThreshold;

    int acols, arows;
    const int *adata;
    Mutex& _lock;
};

static bool CheckDistance(const std::vector<Vec3f> &circles, size_t endIdx, const Vec3f& circle, float minDist2)
{
    bool goodPoint = true;
    for (uint j = 0; j < endIdx; ++j)
    {
        Vec3f pt = circles[j];
        float distX = circle[0] - pt[0], distY = circle[1] - pt[1];
        if (distX * distX + distY * distY < minDist2)
        {
            goodPoint = false;
            break;
        }
    }
    return goodPoint;
}

static void GetCircleCenters(const std::vector<Point> &centers, std::vector<Vec3f> &circles, int acols, float minDist, float dr)
{
    size_t centerCnt = centers.size();
    float minDist2 = minDist * minDist;
    for (size_t i = 0; i < centerCnt; ++i)
    {
        //int center = centers[i];
        //int y = center / acols;
        //int x = center - y * acols;
        Vec3f circle = Vec3f((centers[i].x + 0.5f) * dr, (centers[i].y + 0.5f) * dr, 0);

        bool goodPoint = CheckDistance(circles, circles.size(), circle, minDist2);
        if (goodPoint)
            circles.push_back(circle);
    }
}

static void RemoveOverlaps(std::vector<Vec3f>& circles, float minDist)
{
    float minDist2 = minDist * minDist;
    size_t endIdx = 1;
    for (size_t i = 1; i < circles.size(); ++i)
    {
        Vec3f circle = circles[i];
        if (CheckDistance(circles, endIdx, circle, minDist2))
        {
            circles[endIdx] = circle;
            ++endIdx;
        }
    }
    circles.resize(endIdx);
}

template<class NZPoints>
class HoughCircleEstimateRadiusInvoker : public ParallelLoopBody
{
public:
    HoughCircleEstimateRadiusInvoker(const NZPoints &_nz, int _nzSz, const std::vector<Point> &initCenters, std::vector<EstimatedCircle> &_circlesEst,
                                     int _acols, int _accThreshold, int _minRadius, int _maxRadius,
                                     float _dp, Mutex& _mutex) :
        nz(_nz), nzSz(_nzSz), centers(initCenters), circlesEst(_circlesEst), acols(_acols), accThreshold(_accThreshold),
        minRadius(_minRadius), maxRadius(_maxRadius), dr(_dp), _lock(_mutex)
    {
        minRadius2 = (float)minRadius * minRadius;
        maxRadius2 = (float)maxRadius * maxRadius;
        centerSz = (int)centers.size();
        CV_Assert(nzSz > 0);
    }

    ~HoughCircleEstimateRadiusInvoker() {_lock.unlock();}

protected:
    inline int filterCircles(const Point2f& curCenter, float* ddata) const;

    void operator()(const Range &boundaries) const
    {
        std::vector<EstimatedCircle> circlesLocal;
        const int nBinsPerDr = 10;
        int nBins = cvRound((maxRadius - minRadius)/dr*nBinsPerDr);
        AutoBuffer<int> bins(nBins);
        AutoBuffer<float> distBuf(nzSz), distSqrtBuf(nzSz);
        float *ddata = distBuf;
        float *dSqrtData = distSqrtBuf;

        bool singleThread = (boundaries == Range(0, centerSz));
        int i = boundaries.start;

        // For each found possible center
        // Estimate radius and check support
        for(; i < boundaries.end; ++i)
        {
			int y = centers[i].y;
			int x = centers[i].x;

            //Calculate circle's center in pixels
            Point2f curCenter = Point2f((x + 0.5f) * dr, (y + 0.5f) * dr);
            int nzCount = filterCircles(curCenter, ddata);

            int maxCount = 0;
            float rBest = 0;
            if(nzCount)
            {
                Mat_<float> distMat(1, nzCount, ddata);
                Mat_<float> distSqrtMat(1, nzCount, dSqrtData);
                sqrt(distMat, distSqrtMat);

                memset(bins, 0, sizeof(bins[0])*bins.size());
                for(int k = 0; k < nzCount; k++)
                {
                    int bin = std::max(0, std::min(nBins-1, cvRound((dSqrtData[k] - minRadius)/dr*nBinsPerDr)));
                    bins[bin]++;
                }

                for(int j = nBins - 1; j > 0; j--)
                {
                    if(bins[j])
                    {
                        int upbin = j;
                        int curCount = 0;
                        for(; j > upbin - nBinsPerDr && j >= 0; j--)
                        {
                            curCount += bins[j];
                        }
                        float rCur = (upbin + j)/2.f /nBinsPerDr * dr + minRadius;
                        if((curCount * rBest >= maxCount * rCur) ||
                            (rBest < FLT_EPSILON && curCount >= maxCount))
                        {
                            rBest = rCur;
                            maxCount = curCount;
                        }
                    }
                }
            }

            // Check if the circle has enough support
            if(maxCount > accThreshold)
            {
                circlesLocal.push_back(EstimatedCircle(Vec3f(curCenter.x, curCenter.y, rBest), maxCount));
            }
        }

        if(!circlesLocal.empty())
        {
            std::sort(circlesLocal.begin(), circlesLocal.end(), cmpAccum);
            if(singleThread)
            {
                std::swap(circlesEst, circlesLocal);
            }
            else
            {
                AutoLock alock(_lock);
                if (circlesEst.empty())
                    std::swap(circlesEst, circlesLocal);
                else
                    circlesEst.insert(circlesEst.end(), circlesLocal.begin(), circlesLocal.end());
            }
        }
    }

private:
    const NZPoints &nz;
    int nzSz;
    const std::vector<Point> &centers;
    std::vector<EstimatedCircle> &circlesEst;
    int acols, accThreshold, minRadius, maxRadius;
    float dr;
    int centerSz;
    float minRadius2, maxRadius2;
    Mutex& _lock;
};

template<>
inline int HoughCircleEstimateRadiusInvoker<NZPointList>::filterCircles(const Point2f& curCenter, float* ddata) const
{
    int nzCount = 0;
    const Point* nz_ = &nz[0];
    int j = 0;
#if CV_SIMD128
    {
        const v_float32x4 v_minRadius2 = v_setall_f32(minRadius2);
        const v_float32x4 v_maxRadius2 = v_setall_f32(maxRadius2);

        v_float32x4 v_curCenterX = v_setall_f32(curCenter.x);
        v_float32x4 v_curCenterY = v_setall_f32(curCenter.y);

        float CV_DECL_ALIGNED(16) rbuf[4];
        for(; j <= nzSz - 4; j += 4)
        {
            v_float32x4 v_nzX, v_nzY;
            v_load_deinterleave((const float*)&nz_[j], v_nzX, v_nzY); // FIXIT use proper datatype

            v_float32x4 v_x = v_cvt_f32(v_reinterpret_as_s32(v_nzX));
            v_float32x4 v_y = v_cvt_f32(v_reinterpret_as_s32(v_nzY));

            v_float32x4 v_dx = v_x - v_curCenterX;
            v_float32x4 v_dy = v_y - v_curCenterY;

            v_float32x4 v_r2 = (v_dx * v_dx) + (v_dy * v_dy);
            v_float32x4 vmask = (v_minRadius2 <= v_r2) & (v_r2 <= v_maxRadius2);
            unsigned int mask = v_signmask(vmask);
            if (mask)
            {
                v_store_aligned(rbuf, v_r2);
                if (mask & 1) ddata[nzCount++] = rbuf[0];
                if (mask & 2) ddata[nzCount++] = rbuf[1];
                if (mask & 4) ddata[nzCount++] = rbuf[2];
                if (mask & 8) ddata[nzCount++] = rbuf[3];
            }
        }
    }
#endif

    // Estimate best radius
    for(; j < nzSz; ++j)
    {
        const Point pt = nz_[j];
        float _dx = curCenter.x - pt.x, _dy = curCenter.y - pt.y;
        float _r2 = _dx * _dx + _dy * _dy;

        if(minRadius2 <= _r2 && _r2 <= maxRadius2)
        {
            ddata[nzCount++] = _r2;
        }
    }
    return nzCount;
}

template<>
inline int HoughCircleEstimateRadiusInvoker<NZPointSet>::filterCircles(const Point2f& curCenter, float* ddata) const
{
    int nzCount = 0;
    const Mat_<uchar>& positions = nz.positions;

    const int rOuter = maxRadius + 1;
    const Range xOuter = Range(std::max(int(curCenter.x - rOuter), 0), std::min(int(curCenter.x + rOuter), positions.cols));
    const Range yOuter = Range(std::max(int(curCenter.y - rOuter), 0), std::min(int(curCenter.y + rOuter), positions.rows));

#if CV_SIMD128
    const int numSIMDPoints = 4;

    const v_float32x4 v_minRadius2 = v_setall_f32(minRadius2);
    const v_float32x4 v_maxRadius2 = v_setall_f32(maxRadius2);
    const v_float32x4 v_curCenterX_0123 = v_setall_f32(curCenter.x) - v_float32x4(0.0f, 1.0f, 2.0f, 3.0f);
#endif

    for (int y = yOuter.start; y < yOuter.end; y++)
    {
        const uchar* ptr = positions.ptr(y, 0);
        float dy = curCenter.y - y;
        float dy2 = dy * dy;

        int x = xOuter.start;
#if CV_SIMD128
        {
            const v_float32x4 v_dy2 = v_setall_f32(dy2);
            const v_uint32x4 v_zero_u32 = v_setall_u32(0);
            float CV_DECL_ALIGNED(16) rbuf[4];
            for (; x <= xOuter.end - 4; x += numSIMDPoints)
            {
                v_uint32x4 v_mask = v_load_expand_q(ptr + x);
                v_mask = v_mask != v_zero_u32;

                v_float32x4 v_x = v_cvt_f32(v_setall_s32(x));
                v_float32x4 v_dx = v_x - v_curCenterX_0123;

                v_float32x4 v_r2 = (v_dx * v_dx) + v_dy2;
                v_float32x4 vmask = (v_minRadius2 <= v_r2) & (v_r2 <= v_maxRadius2) & v_reinterpret_as_f32(v_mask);
                unsigned int mask = v_signmask(vmask);
                if (mask)
                {
                    v_store_aligned(rbuf, v_r2);
                    if (mask & 1) ddata[nzCount++] = rbuf[0];
                    if (mask & 2) ddata[nzCount++] = rbuf[1];
                    if (mask & 4) ddata[nzCount++] = rbuf[2];
                    if (mask & 8) ddata[nzCount++] = rbuf[3];
                }
            }
        }
#endif
        for (; x < xOuter.end; x++)
        {
            if (ptr[x])
            {
                float _dx = curCenter.x - x;
                float _r2 = _dx * _dx + dy2;
                if(minRadius2 <= _r2 && _r2 <= maxRadius2)
                {
                    ddata[nzCount++] = _r2;
                }
            }
        }
    }
    return nzCount;
}

static void HoughCirclesGradient(InputArray _image, OutputArray _circles, std::vector<Point>& initCenters, float dp, float minDist,
                                 int minRadius, int maxRadius, int cannyThreshold,
                                 int accThreshold, int maxCircles, int kernelSize, bool centersOnly)
{
    CV_Assert(kernelSize == -1 || kernelSize == 3 || kernelSize == 5 || kernelSize == 7);
    dp = max(dp, 1.f);
    float idp = 1.f/dp;

    Mat edges, dx, dy;

    Sobel(_image, dx, CV_16S, 1, 0, kernelSize, 1, 0, BORDER_REPLICATE);
    Sobel(_image, dy, CV_16S, 0, 1, kernelSize, 1, 0, BORDER_REPLICATE);
    Canny(dx, dy, edges, std::max(1, cannyThreshold / 2), cannyThreshold, false);

    Mutex mtx;
    int numThreads = std::max(1, getNumThreads());
    std::vector<Mat> accumVec;
    NZPointSet nz(_image.rows(), _image.cols());
    parallel_for_(Range(0, edges.rows),
                  HoughCirclesAccumInvoker(edges, dx, dy, minRadius, maxRadius, idp, accumVec, nz, mtx),
                  numThreads);
    int nzSz = cv::countNonZero(nz.positions);
    if(nzSz <= 0)
        return;

    Mat accum = accumVec[0];
    for(size_t i = 1; i < accumVec.size(); i++)
    {
        accum += accumVec[i];
    }
    accumVec.clear();

    //std::vector<int> centers;

    // 4 rows when multithreaded because there is a bit overhead
    // and on the other side there are some row ranges where centers are concentrated
    /*parallel_for_(Range(1, accum.rows - 1),
                  HoughCirclesFindCentersInvoker(accum, initCenters, accThreshold, mtx),
                  (numThreads > 1) ? ((accum.rows - 2) / 4) : 1);*/
	//Find possible circle centers
	const int* adata = accum.ptr<int>();
	std::vector<cv::Point> filteredCenters;
	for (size_t i = 0; i < initCenters.size(); i++)
	{
		int blockSize = 10;
		for (int j = -blockSize; j < blockSize; j+=2)
		{
			int y = initCenters[i].y + j;
			for (int k = -blockSize; k < blockSize; k += 2)
			{
				int x = initCenters[i].x + k;
				if (x <= 0 || y <= 0 || x >= accum.cols -1 || y >= accum.rows -1)
				{
					continue;
				}
				int base = y * accum.cols + x;
				int tmp1 = adata[base];
				int tmp2 = adata[base - 1];
				int tmp3 = adata[base - accum.cols];
				int tmp4 = adata[base + 1];
				int tmp5 = adata[base + accum.cols];
				if (adata[base] > accThreshold &&
						adata[base] > adata[base - 1] && adata[base] >= adata[base + 1] &&
						adata[base] > adata[base - accum.cols] && adata[base] >= adata[base + accum.cols])
				{
					filteredCenters.push_back(cv::Point(x, y));
				}
				
			}
		}
	}
	initCenters = filteredCenters;
    int centerCnt = (int)initCenters.size();
    if(centerCnt == 0)
        return;

    //std::sort(initCenters.begin(), initCenters.end(), hough_cmp_gt(accum.ptr<Point>()));

    std::vector<Vec3f> circles;
    circles.reserve(256);
    if (centersOnly)
    {
        // Just get the circle centers
        GetCircleCenters(initCenters, circles, accum.cols, minDist, dp);
    }
    else
    {
        std::vector<EstimatedCircle> circlesEst;
        if (nzSz < maxRadius * maxRadius)
        {
            // Faster to use a list
            NZPointList nzList(nzSz);
            nz.toList(nzList);
            // One loop iteration per thread if multithreaded.
            parallel_for_(Range(0, centerCnt),
                HoughCircleEstimateRadiusInvoker<NZPointList>(nzList, nzSz, initCenters, circlesEst, accum.cols,
                    accThreshold, minRadius, maxRadius, dp, mtx),
                numThreads);
        }
        else
        {
            // Faster to use a matrix
            // One loop iteration per thread if multithreaded.
            parallel_for_(Range(0, centerCnt),
                HoughCircleEstimateRadiusInvoker<NZPointSet>(nz, nzSz, initCenters, circlesEst, accum.cols,
                    accThreshold, minRadius, maxRadius, dp, mtx),
                numThreads);
        }

        // Sort by accumulator value
        std::sort(circlesEst.begin(), circlesEst.end(), cmpAccum);
        std::transform(circlesEst.begin(), circlesEst.end(), std::back_inserter(circles), GetCircle);
        RemoveOverlaps(circles, minDist);
    }

    if(circles.size() > 0)
    {
        int numCircles = std::min(maxCircles, int(circles.size()));
        _circles.create(1, numCircles, CV_32FC3);
        Mat(1, numCircles, CV_32FC3, &circles[0]).copyTo(_circles.getMat());
        return;
    }
}

static void HoughCircles( InputArray _image, OutputArray _circles, std::vector<Point> &initCenters,
                          int method, double dp, double minDist,
                          double param1, double param2,
                          int minRadius, int maxRadius,
                          int maxCircles, double param3 )
{
    CV_INSTRUMENT_REGION()

    CV_Assert(!_image.empty() && _image.type() == CV_8UC1 && (_image.isMat() || _image.isUMat()));
    CV_Assert(_circles.isMat() || _circles.isVector());

    if( dp <= 0 || minDist <= 0 || param1 <= 0 || param2 <= 0)
        CV_Error( Error::StsOutOfRange, "dp, min_dist, canny_threshold and acc_threshold must be all positive numbers" );

    int cannyThresh = cvRound(param1), accThresh = cvRound(param2), kernelSize = cvRound(param3);

    minRadius = std::max(0, minRadius);

    if(maxCircles < 0)
        maxCircles = INT_MAX;

    bool centersOnly = (maxRadius < 0);

    if( maxRadius <= 0 )
        maxRadius = std::max( _image.rows(), _image.cols() );
    else if( maxRadius <= minRadius )
        maxRadius = minRadius + 2;

    switch( method )
    {
    case CV_HOUGH_GRADIENT:
        HoughCirclesGradient(_image, _circles, initCenters, (float)dp, (float)minDist,
                             minRadius, maxRadius, cannyThresh,
                             accThresh, maxCircles, kernelSize, centersOnly);
        break;
    default:
        CV_Error( Error::StsBadArg, "Unrecognized method id. Actually only CV_HOUGH_GRADIENT is supported." );
    }
}

void HoughCircles( InputArray _image, OutputArray _circles, std::vector<Point>& initCenters, 
                   int method, double dp, double minDist,
                   double param1, double param2,
                   int minRadius, int maxRadius )
{
    HoughCircles(_image, _circles, initCenters, method, dp, minDist, param1, param2, minRadius, maxRadius, -1, 3);
}
} // \namespace cv

/* Wrapper function for standard hough transform */
//CV_IMPL CvSeq*
//cvHoughLines2( CvArr* src_image, void* lineStorage, int method,
//               double rho, double theta, int threshold,
//               double param1, double param2,
//               double min_theta, double max_theta )
//{
//    cv::Mat image = cv::cvarrToMat(src_image);
//    std::vector<cv::Vec2f> l2;
//    std::vector<cv::Vec4i> l4;
//
//    CvMat* mat = 0;
//    CvSeq* lines = 0;
//    CvSeq lines_header;
//    CvSeqBlock lines_block;
//    int lineType, elemSize;
//    int linesMax = INT_MAX;
//    int iparam1, iparam2;
//
//    if( !lineStorage )
//        CV_Error(cv::Error::StsNullPtr, "NULL destination" );
//
//    if( rho <= 0 || theta <= 0 || threshold <= 0 )
//        CV_Error( cv::Error::StsOutOfRange, "rho, theta and threshold must be positive" );
//
//    if( method != CV_HOUGH_PROBABILISTIC )
//    {
//        lineType = CV_32FC2;
//        elemSize = sizeof(float)*2;
//    }
//    else
//    {
//        lineType = CV_32SC4;
//        elemSize = sizeof(int)*4;
//    }
//
//    bool isStorage = isStorageOrMat(lineStorage);
//
//    if( isStorage )
//    {
//        lines = cvCreateSeq( lineType, sizeof(CvSeq), elemSize, (CvMemStorage*)lineStorage );
//    }
//    else
//    {
//        mat = (CvMat*)lineStorage;
//
//        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) )
//            CV_Error( CV_StsBadArg,
//            "The destination matrix should be continuous and have a single row or a single column" );
//
//        if( CV_MAT_TYPE( mat->type ) != lineType )
//            CV_Error( CV_StsBadArg,
//            "The destination matrix data type is inappropriate, see the manual" );
//
//        lines = cvMakeSeqHeaderForArray( lineType, sizeof(CvSeq), elemSize, mat->data.ptr,
//                                         mat->rows + mat->cols - 1, &lines_header, &lines_block );
//        linesMax = lines->total;
//        cvClearSeq( lines );
//    }
//
//    iparam1 = cvRound(param1);
//    iparam2 = cvRound(param2);
//
//    switch( method )
//    {
//    case CV_HOUGH_STANDARD:
//        HoughLinesStandard( image, (float)rho,
//                (float)theta, threshold, l2, linesMax, min_theta, max_theta );
//        break;
//    case CV_HOUGH_MULTI_SCALE:
//        HoughLinesSDiv( image, (float)rho, (float)theta,
//                threshold, iparam1, iparam2, l2, linesMax, min_theta, max_theta );
//        break;
//    case CV_HOUGH_PROBABILISTIC:
//        HoughLinesProbabilistic( image, (float)rho, (float)theta,
//                threshold, iparam1, iparam2, l4, linesMax );
//        break;
//    default:
//        CV_Error( CV_StsBadArg, "Unrecognized method id" );
//    }
//
//    int nlines = (int)(l2.size() + l4.size());
//
//    if( !isStorage )
//    {
//        if( mat->cols > mat->rows )
//            mat->cols = nlines;
//        else
//            mat->rows = nlines;
//    }
//
//    if( nlines )
//    {
//        cv::Mat lx = method == CV_HOUGH_STANDARD || method == CV_HOUGH_MULTI_SCALE ?
//            cv::Mat(nlines, 1, CV_32FC2, &l2[0]) : cv::Mat(nlines, 1, CV_32SC4, &l4[0]);
//
//        if (isStorage)
//        {
//            cvSeqPushMulti(lines, lx.ptr(), nlines);
//        }
//        else
//        {
//            cv::Mat dst(nlines, 1, lx.type(), mat->data.ptr);
//            lx.copyTo(dst);
//        }
//    }
//
//    if( isStorage )
//        return lines;
//    return 0;
//}


CV_IMPL CvSeq*
cvHoughCircles( CvArr* src_image, void* circle_storage, std::vector<cv::Point> initCenters,
                int method, double dp, double min_dist,
                double param1, double param2,
                int min_radius, int max_radius )
{
    CvSeq* circles = NULL;
    int circles_max = INT_MAX;
    cv::Mat src = cv::cvarrToMat(src_image), circles_mat;

    if( !circle_storage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    bool isStorage = isStorageOrMat(circle_storage);

    if(isStorage)
    {
        circles = cvCreateSeq( CV_32FC3, sizeof(CvSeq),
            sizeof(float)*3, (CvMemStorage*)circle_storage );
    }
    else
    {
        CvSeq circles_header;
        CvSeqBlock circles_block;
        CvMat *mat = (CvMat*)circle_storage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) ||
            CV_MAT_TYPE(mat->type) != CV_32FC3 )
            CV_Error( CV_StsBadArg,
                      "The destination matrix should be continuous and have a single row or a single column" );

        circles = cvMakeSeqHeaderForArray( CV_32FC3, sizeof(CvSeq), sizeof(float)*3,
                mat->data.ptr, mat->rows + mat->cols - 1, &circles_header, &circles_block );
        circles_max = circles->total;
        cvClearSeq( circles );
    }

    cv::HoughCircles(src, circles_mat, initCenters, method, dp, min_dist, param1, param2, min_radius, max_radius, circles_max, 3);
    cvSeqPushMulti(circles, circles_mat.data, (int)circles_mat.total());
    return circles;
}

/* End of file. */
