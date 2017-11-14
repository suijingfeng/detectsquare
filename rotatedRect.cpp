#include <cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#define SAVE_TEMP_RES   1

using namespace std;

cv::Mat g_img;

enum { CALIPERS_MAXHEIGHT=0, CALIPERS_MINAREARECT=1, CALIPERS_MAXDIST=2 };


/* Parameters:
	 points      - convex hull vertices (any orientation)
	 n           - number of vertices
	 mode        - concrete application of algorithm can be  CV_CALIPERS_MAXDIST or CV_CALIPERS_MINAREARECT
	 left, bottom, right, top - indexes of extremal points
	 out         - output info.
	 In case CV_CALIPERS_MAXDIST it points to float value -
	   maximal height of polygon.
	 In case CV_CALIPERS_MINAREARECT
	 ((CvPoint2D32f*)out)[0] - corner
	 ((CvPoint2D32f*)out)[1] - vector1
	 ((CvPoint2D32f*)out)[0] - corner2
	 //
	 //                      ^
	 //                      |
	 //              vector2 |
	 //                      |
	 //                      |____________\
	 //                    corner         /
	 //                               vector1
    we will use usual cartesian coordinates 
*/


void rotatingCalipers(std::vector< cv::Point > &points, int n, int mode, float* out)
{
    float minarea = FLT_MAX, max_dist = 0;
    char buffer[32] = {};
    cv::AutoBuffer<float> abuf(n*3);
    float* inv_vect_length = abuf;
    cv::Point2f* vect = (cv::Point2f*)(inv_vect_length + n);
    int left = 0, bottom = 0, right = 0, top = 0;
    int seq[4] = {-1, -1, -1, -1};
    int i, k;

    /* rotating calipers sides will always have coordinates (a,b) (-b,a) (-a,-b) (b, -a) */
    /* this is a first base bector (a,b) initialized by (1,0) */
    float orientation = 0;
    float base_a;
    float base_b = 0;

    float left_x, right_x, top_y, bottom_y;
    cv::Point2f pt0 = points[0];

    left_x = right_x = pt0.x;
    top_y = bottom_y = pt0.y;

    for( i = 0; i < n; i++ )
    {
        double dx, dy;

        if( pt0.x < left_x )
            left_x = pt0.x, left = i;

        if( pt0.x > right_x )
            right_x = pt0.x, right = i;

        if( pt0.y > top_y )
            top_y = pt0.y, top = i;

        if( pt0.y < bottom_y )
            bottom_y = pt0.y, bottom = i;

        cv::Point2f pt = points[(i+1) & (i+1 < n ? -1 : 0)];

        dx = pt.x - pt0.x;
        dy = pt.y - pt0.y;

        vect[i].x = (float)dx;
        vect[i].y = (float)dy;
        inv_vect_length[i] = (float)(1./std::sqrt(dx*dx + dy*dy));

        pt0 = pt;
    }

    // find convex hull orientation
    {
        double ax = vect[n-1].x;
        double ay = vect[n-1].y;

        for( i = 0; i < n; i++ )
        {
            double bx = vect[i].x;
            double by = vect[i].y;

            double convexity = ax * by - ay * bx;

            if( convexity != 0 )
            {
                orientation = (convexity > 0) ? 1.f : (-1.f);
                break;
            }
            ax = bx;
            ay = by;
        }
        CV_Assert( orientation != 0 );
    }
    base_a = orientation;

    /*****************************************************************************************/
    /*                         init calipers position                                        */
    seq[0] = bottom;
    seq[1] = right;
    seq[2] = top;
    seq[3] = left;
    /*****************************************************************************************/
    /*                         Main loop - evaluate angles and rotate calipers               */

    /* all of edges will be checked while rotating calipers by 90 degrees */
    for(k = 0; k < n; k++ )
    {
        /* sinus of minimal angle */
        /*float sinus;*/

        /* compute cosine of angle between calipers side and polygon edge */
        /* dp - dot product */
        float dp[4] =
        {
            +base_a * vect[seq[0]].x + base_b * vect[seq[0]].y,
            -base_b * vect[seq[1]].x + base_a * vect[seq[1]].y,
            -base_a * vect[seq[2]].x - base_b * vect[seq[2]].y,
            +base_b * vect[seq[3]].x - base_a * vect[seq[3]].y,
        };

        float maxcos = dp[0] * inv_vect_length[seq[0]];

        /* number of calipers edges, that has minimal angle with edge */
        int main_element = 0;

        /* choose minimal angle */
        for(i = 1; i < 4; ++i )
        {
            float cosalpha = dp[i] * inv_vect_length[seq[i]];
            if(cosalpha > maxcos)
            {
                main_element = i;
                maxcos = cosalpha;
            }
        }

        /*rotate calipers*/
        {
            //get next base
            int pindex = seq[main_element];
            float lead_x = vect[pindex].x * inv_vect_length[pindex];
            float lead_y = vect[pindex].y * inv_vect_length[pindex];
            switch( main_element )
            {
                case 0:
                    base_a = lead_x;
                    base_b = lead_y;
                    break;
                case 1:
                    base_a = lead_y;
                    base_b = -lead_x;
                    break;
                case 2:
                    base_a = -lead_x;
                    base_b = -lead_y;
                    break;
                case 3:
                    base_a = -lead_y;
                    base_b = lead_x;
                    break;
                default:
                    CV_Error(CV_StsError, "main_element should be 0, 1, 2 or 3");
            }
        }
        /* change base point of main edge */
        seq[main_element] += 1;
        seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];

        switch(mode)
        {
            case CALIPERS_MAXHEIGHT:
            {
                /* now main element lies on edge alligned to calipers side */
                /* find opposite element i.e. transform  */
                /* 0->2, 1->3, 2->0, 3->1                */
                int opposite_el = main_element ^ 2;

                float dx = points[seq[opposite_el]].x - points[seq[main_element]].x;
                float dy = points[seq[opposite_el]].y - points[seq[main_element]].y;
                float dist;

                if( main_element & 1 )
                    dist = (float)fabs(dx * base_a + dy * base_b);
                else
                    dist = (float)fabs(dx * (-base_b) + dy * base_a);

                if( dist > max_dist )
                    max_dist = dist;
            }break;
            case CALIPERS_MINAREARECT:
            { /* find area of rectangle */

                float height;
                float area;

                /* find vector left-right */
                float dx = points[seq[1]].x - points[seq[3]].x;
                float dy = points[seq[1]].y - points[seq[3]].y;

                /* dotproduct */
                float width = dx * base_a + dy * base_b;

                /* find vector left-right */
                dx = points[seq[2]].x - points[seq[0]].x;
                dy = points[seq[2]].y - points[seq[0]].y;

                /* dotproduct */
                height = -dx * base_b + dy * base_a;

                area = width * height;
                if( area <= minarea )
                {
                    float *buf = (float *) buffer;

                    minarea = area;
                    /* leftist point */
                    ((int *) buf)[0] = seq[3];
                    buf[1] = base_a;
                    buf[2] = width;
                    buf[3] = base_b;
                    buf[4] = height;
                    /* bottom point */
                    ((int *) buf)[5] = seq[0];
                    buf[6] = area;
                }
            }break;
        } /* switch ended */
    }     /* for ended */

    switch(mode)
    {
        case CALIPERS_MINAREARECT:
        {
            float *buf = (float *)buffer;

            float A1 = buf[1];
            float B1 = buf[3];

            float A2 = -buf[3];
            float B2 = buf[1];

            float C1 = A1 * points[((int *) buf)[0]].x + points[((int *) buf)[0]].y * B1;
            float C2 = A2 * points[((int *) buf)[5]].x + points[((int *) buf)[5]].y * B2;

            float idet = 1.f / (A1 * B2 - A2 * B1);

            float px = (C1 * B2 - C2 * B1) * idet;
            float py = (A1 * C2 - A2 * C1) * idet;

            out[0] = px;
            out[1] = py;

            out[2] = A1 * buf[2];
            out[3] = B1 * buf[2];

            out[4] = A2 * buf[4];
            out[5] = B2 * buf[4];
        }break;
        case CALIPERS_MAXHEIGHT:
        {
            out[0] = max_dist;
        }break;
    }
}


cv::RotatedRect minRectArea(std::vector<cv::Point> _points)
{
    std::vector<cv::Point> hpoints;

    cv::Point2f out[3];
    cv::RotatedRect box;

    convexHull(_points, hpoints, true, true);
    int n = hpoints.size();

    if( n > 2 )
    {
        rotatingCalipers(hpoints, n, CALIPERS_MINAREARECT, (float*)out );
        box.center.x = out[0].x + (out[1].x + out[2].x)*0.5f;
        box.center.y = out[0].y + (out[1].y + out[2].y)*0.5f;
        box.size.width = (float)std::sqrt((double)out[1].x*out[1].x + (double)out[1].y*out[1].y);
        box.size.height = (float)std::sqrt((double)out[2].x*out[2].x + (double)out[2].y*out[2].y);
        box.angle = (float)atan2( (double)out[1].y, (double)out[1].x );
    }
    else if( n == 2 )
    {
        box.center.x = (hpoints[0].x + hpoints[1].x)*0.5f;
        box.center.y = (hpoints[0].y + hpoints[1].y)*0.5f;
        double dx = hpoints[1].x - hpoints[0].x;
        double dy = hpoints[1].y - hpoints[0].y;
        box.size.width = (float)std::sqrt(dx*dx + dy*dy);
        box.size.height = 0;
        box.angle = (float)atan2( dy, dx );
    }
    else if( n == 1 )
    {
        box.center = hpoints[0];
    }
    
    box.angle = (float)(box.angle*180/CV_PI);

    return box;
}


void drawBox(cv::Mat &img, cv::RotatedRect box)
{
    if (img.channels() == 1)
        cvtColor(img, img, cv::COLOR_GRAY2BGR);
    
    cv::Point2f vertices[4];
    box.points(vertices);
    
    //for (int i = 0; i < 4; i++)
    //    line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));
    
    line(img, vertices[0], vertices[1], cv::Scalar(0,255,0), 2);
    line(img, vertices[1], vertices[2], cv::Scalar(0,255,0), 2);
    line(img, vertices[2], vertices[3], cv::Scalar(0,255,0), 2);
    line(img, vertices[3], vertices[0], cv::Scalar(0,255,0), 2);

    imshow("rectangles", img);
    cv::waitKey(0);
}


void GetBlackQuadHypotheses(const std::vector<std::vector< cv::Point > > & contours, const std::vector< cv::Vec4i > & hierarchy, std::vector<float> & quads)
{
    const float min_aspect_ratio = 0.5f, max_aspect_ratio = 2.0f;
    const float min_quad_size = 10.0f, max_quad_pixel = 250;
    
    std::vector< std::vector< cv::Point > >::const_iterator pt;
    std::cout << "contours.size(): " << contours.size() << std::endl;

    for(pt = contours.begin(); pt != contours.end(); ++pt)
    {
        const std::vector< std::vector< cv::Point > >::const_iterator::difference_type idx = pt - contours.begin();
        
        // if the parent contour not equal to -1, then it is a hole
        //if(hierarchy.at(idx)[3] != -1)
        if(hierarchy[idx][2] != -1)
            continue; // skip holes
        
        // too small, min_box_size * 4
        if(pt->size() < 20)
            continue;
        //const std::vector< cv::Point > & c = *i;
        std::cout << "pt->size(): " << pt->size() << std::endl;
        
        cv::RotatedRect box = minRectArea(*pt);

        float box_size = MAX(box.size.width, box.size.height);
        
        if(box_size < min_quad_size || box_size > max_quad_pixel)
            continue;

        float aspect_ratio = box.size.width / MAX(box.size.height, 1);
        if(aspect_ratio < min_aspect_ratio || aspect_ratio > max_aspect_ratio)
            continue;
        
        drawBox(g_img, box);
        quads.push_back( box_size );
    }
    std::cout << "quads.size(): " << quads.size() << std::endl;
}



bool checkBlackQuads(std::vector<float> & quads, const cv::Size & size)
{
    const unsigned int black_quads_count = (size.width+1) * (size.height+1) / 2 + 1;
    const unsigned int min_quads_count = black_quads_count * 0.75;
    const float quad_aspect_ratio = 1.4f;

    std::sort(quads.begin(), quads.end());
    
    // now check if there are many hypotheses with similar sizes
    for(unsigned int i = 0; i < quads.size(); i++)
    {
        unsigned int j = i + 1;
        for(; j < quads.size(); j++)
            if(quads[j]/quads[i] > quad_aspect_ratio)
                break;

        if(j - i + 1 > min_quads_count )
            return true; // check the number of black squares
    }
    return false;
}


struct CvLinkedRunPoint
{
    struct CvLinkedRunPoint* link;
    struct CvLinkedRunPoint* next;
    CvPoint pt;
};

static inline cv::Size cvGetMatSize( const CvMat* mat )
{
    return cv::Size(mat->cols, mat->rows);
}

#define CV_GET_WRITTEN_ELEM( writer ) ((writer).ptr - (writer).seq->elem_size)
#define ICV_SINGLE              0
#define ICV_CONNECTING_ABOVE    1
#define ICV_CONNECTING_BELOW    -1

#if CV_SSE2
static inline unsigned int trailingZeros(unsigned int value)
{
    CV_DbgAssert(value != 0); // undefined for zero input (https://en.wikipedia.org/wiki/Find_first_set)
    #if defined(_MSC_VER)
        #if (_MSC_VER < 1700)
            unsigned long index = 0;
            _BitScanForward(&index, value);
            return (unsigned int)index;
        #else
    inline int findEndContourPoint(uchar *src_data, CvSize img_size, int j, bool haveSIMD) {
#if CV_SSE2
    if (j < img_size.width && !src_data[j]) {
        return j;
    } else if (haveSIMD) {
        __m128i v_zero = _mm_setzero_si128();
        int v_size = img_size.width - 32;

        for (; j <= v_size; j += 32) {
            __m128i v_p1 = _mm_loadu_si128((const __m128i*)(src_data + j));
            __m128i v_p2 = _mm_loadu_si128((const __m128i*)(src_data + j + 16));

            __m128i v_cmp1 = _mm_cmpeq_epi8(v_p1, v_zero);
            __m128i v_cmp2 = _mm_cmpeq_epi8(v_p2, v_zero);

            unsigned int mask1 = _mm_movemask_epi8(v_cmp1);
            unsigned int mask2 = _mm_movemask_epi8(v_cmp2);

            if (mask1) {
                j += trailingZeros(mask1);
                return j;
            }

            if (mask2) {
                j += trailingZeros(mask2 << 16);
                return j;
            }
        }

        if (j <= img_size.width - 16) {
            __m128i v_p = _mm_loadu_si128((const __m128i*)(src_data + j));

            unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v_p, v_zero));

            if (mask) {
                j += trailingZeros(mask);
                return j;
            }
            j += 16;
        }
    }
#else
    CV_UNUSED(haveSIMD);
#endif
    for (; j < img_size.width && src_data[j]; ++j)
        ;

    return j;
}        return _tzcnt_u32(value);
        #endif
    #elif defined(__GNUC__) || defined(__GNUG__)
        return __builtin_ctz(value);
    #elif defined(__ICC) || defined(__INTEL_COMPILER)
        return _bit_scan_forward(value);
    #elif defined(__clang__)
        return llvm.cttz.i32(value, true);
    #else
        static const int MultiplyDeBruijnBitPosition[32] = {
            0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
            31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9 };
        return MultiplyDeBruijnBitPosition[((uint32_t)((value & -value) * 0x077CB531U)) >> 27];
    #endif
}
#endif


inline int findStartContourPoint(uchar *src_data, CvSize img_size, int j, bool haveSIMD)
{
#if CV_SSE2
    if (haveSIMD)
    {
        __m128i v_zero = _mm_setzero_si128();
        int v_size = img_size.width - 32;

        for (; j <= v_size; j += 32) {
            __m128i v_p1 = _mm_loadu_si128((const __m128i*)(src_data + j));
            __m128i v_p2 = _mm_loadu_si128((const __m128i*)(src_data + j + 16));

            __m128i v_cmp1 = _mm_cmpeq_epi8(v_p1, v_zero);
            __m128i v_cmp2 = _mm_cmpeq_epi8(v_p2, v_zero);

            unsigned int mask1 = _mm_movemask_epi8(v_cmp1);
            unsigned int mask2 = _mm_movemask_epi8(v_cmp2);

            mask1 ^= 0x0000ffff;
            mask2 ^= 0x0000ffff;

            if (mask1) {
                j += trailingZeros(mask1);
                return j;
            }

            if (mask2) {
                j += trailingZeros(mask2 << 16);
                return j;
            }
        }

        if (j <= img_size.width - 16)
        {
            __m128i v_p = _mm_loadu_si128((const __m128i*)(src_data + j));

            unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v_p, v_zero)) ^ 0x0000ffff;

            if (mask) {
                j += trailingZeros(mask);
                return j;
            }
            j += 16;
        }
    }
#else
    CV_UNUSED(haveSIMD);
#endif
    for(; j < img_size.width && !src_data[j]; ++j)
        ;
    return j;
}


inline int findEndContourPoint(unsigned char *src_data, CvSize img_size, int j, bool haveSIMD)
{
#if CV_SSE2
    if (j < img_size.width && !src_data[j])
        return j;
    else if (haveSIMD)
    {
        __m128i v_zero = _mm_setzero_si128();
        int v_size = img_size.width - 32;

        for(; j <= v_size; j += 32)
        {
            __m128i v_p1 = _mm_loadu_si128((const __m128i*)(src_data + j));
            __m128i v_p2 = _mm_loadu_si128((const __m128i*)(src_data + j + 16));

            __m128i v_cmp1 = _mm_cmpeq_epi8(v_p1, v_zero);
            __m128i v_cmp2 = _mm_cmpeq_epi8(v_p2, v_zero);

            unsigned int mask1 = _mm_movemask_epi8(v_cmp1);
            unsigned int mask2 = _mm_movemask_epi8(v_cmp2);

            if(mask1)
            {
                j += trailingZeros(mask1);
                return j;
            }

            if(mask2)
            {
                j += trailingZeros(mask2 << 16);
                return j;
            }
        }

        if(j <= img_size.width - 16)
        {
            __m128i v_p = _mm_loadu_si128((const __m128i*)(src_data + j));

            unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v_p, v_zero));

            if (mask)
            {
                j += trailingZeros(mask);
                return j;
            }
            j += 16;
        }
    }
#else
    CV_UNUSED(haveSIMD);
#endif
    for (; j < img_size.width && src_data[j]; ++j)
        ;

    return j;
}


static int icvFindContoursInInterval(const CvArr* src, CvMemStorage* storage, CvSeq** result, int contourHeaderSize )
{
    int count = 0;
    cv::Ptr<CvMemStorage> storage00;
    cv::Ptr<CvMemStorage> storage01;
    CvSeq* first = 0;

    int i, j, k, n;

    uchar*  src_data = 0;
    int  img_step = 0;
    CvSize  img_size;

    int  connect_flag;
    int  lower_total;
    int  upper_total;
    int  all_total;
    bool haveSIMD = false;

    CvSeq* runs;
    struct CvLinkedRunPoint  tmp;
    struct CvLinkedRunPoint*  tmp_prev;
    struct CvLinkedRunPoint*  upper_line = 0;
    struct CvLinkedRunPoint*  lower_line = 0;
    struct CvLinkedRunPoint*  last_elem;

    struct CvLinkedRunPoint*  upper_run = 0;
    struct CvLinkedRunPoint*  lower_run = 0;
    struct CvLinkedRunPoint*  prev_point = 0;

    CvSeqWriter  writer_ext;
    CvSeqWriter  writer_int;
    CvSeqWriter  writer;
    CvSeqReader  reader;

    CvSeq* external_contours;
    CvSeq* internal_contours;
    CvSeq* prev = 0;

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    if( !result )
        CV_Error( CV_StsNullPtr, "NULL double CvSeq pointer" );

    if( contourHeaderSize < (int)sizeof(CvContour))
        CV_Error( CV_StsBadSize, "Contour header size must be >= sizeof(CvContour)" );
#if CV_SSE2
    haveSIMD = cv::checkHardwareSupport(CPU_SSE2);
#endif
    storage00.reset(cvCreateChildMemStorage(storage));
    storage01.reset(cvCreateChildMemStorage(storage));

    CvMat stub, *mat;

    mat = cvGetMat( src, &stub );
    if( !CV_IS_MASK_ARR(mat))
        CV_Error( CV_StsBadArg, "Input array must be 8uC1 or 8sC1" );
    src_data = mat->data.ptr;
    img_step = mat->step;
    img_size = cvGetMatSize( mat );

    // Create temporary sequences
    runs = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvLinkedRunPoint), storage00 );
    cvStartAppendToSeq( runs, &writer );

    cvStartWriteSeq( 0, sizeof(CvSeq), sizeof(CvLinkedRunPoint*), storage01, &writer_ext );
    cvStartWriteSeq( 0, sizeof(CvSeq), sizeof(CvLinkedRunPoint*), storage01, &writer_int );

    tmp_prev = &(tmp);
    tmp_prev->next = 0;
    tmp_prev->link = 0;

    // First line. None of runs is binded
    tmp.pt.y = 0;
    i = 0;
    CV_WRITE_SEQ_ELEM( tmp, writer );
    upper_line = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );

    tmp_prev = upper_line;
    for( j = 0; j < img_size.width; )
    {
        j = findStartContourPoint(src_data, img_size, j, haveSIMD);

        if( j == img_size.width )
            break;

        tmp.pt.x = j;
        CV_WRITE_SEQ_ELEM( tmp, writer );
        tmp_prev->next = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );
        tmp_prev = tmp_prev->next;

        j = findEndContourPoint(src_data, img_size, j + 1, haveSIMD);

        tmp.pt.x = j - 1;
        CV_WRITE_SEQ_ELEM( tmp, writer );
        tmp_prev->next = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );
        tmp_prev->link = tmp_prev->next;
        // First point of contour
        CV_WRITE_SEQ_ELEM( tmp_prev, writer_ext );
        tmp_prev = tmp_prev->next;
    }
    cvFlushSeqWriter( &writer );
    upper_line = upper_line->next;
    upper_total = runs->total - 1;
    last_elem = tmp_prev;
    tmp_prev->next = 0;

    for( i = 1; i < img_size.height; i++ )
    {
//------// Find runs in next line
        src_data += img_step;
        tmp.pt.y = i;
        all_total = runs->total;
        for( j = 0; j < img_size.width; )
        {
            j = findStartContourPoint(src_data, img_size, j, haveSIMD);

            if( j == img_size.width ) break;

            tmp.pt.x = j;
            CV_WRITE_SEQ_ELEM( tmp, writer );
            tmp_prev->next = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );
            tmp_prev = tmp_prev->next;

            j = findEndContourPoint(src_data, img_size, j + 1, haveSIMD);

            tmp.pt.x = j - 1;
            CV_WRITE_SEQ_ELEM( tmp, writer );
            tmp_prev = tmp_prev->next = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );
        }//j
        cvFlushSeqWriter( &writer );
        lower_line = last_elem->next;
        lower_total = runs->total - all_total;
        last_elem = tmp_prev;
        tmp_prev->next = 0;
//------//
//------// Find links between runs of lower_line and upper_line
        upper_run = upper_line;
        lower_run = lower_line;
        connect_flag = ICV_SINGLE;

        for( k = 0, n = 0; k < upper_total/2 && n < lower_total/2; )
        {
            switch( connect_flag )
            {
            case ICV_SINGLE:
                if( upper_run->next->pt.x < lower_run->next->pt.x )
                {
                    if( upper_run->next->pt.x >= lower_run->pt.x  -1 )
                    {
                        lower_run->link = upper_run;
                        connect_flag = ICV_CONNECTING_ABOVE;
                        prev_point = upper_run->next;
                    }
                    else
                        upper_run->next->link = upper_run;
                    k++;
                    upper_run = upper_run->next->next;
                }
                else
                {
                    if( upper_run->pt.x <= lower_run->next->pt.x  +1 )
                    {
                        lower_run->link = upper_run;
                        connect_flag = ICV_CONNECTING_BELOW;
                        prev_point = lower_run->next;
                    }
                    else
                    {
                        lower_run->link = lower_run->next;
                        // First point of contour
                        CV_WRITE_SEQ_ELEM( lower_run, writer_ext );
                    }
                    n++;
                    lower_run = lower_run->next->next;
                }
                break;
            case ICV_CONNECTING_ABOVE:
                if( upper_run->pt.x > lower_run->next->pt.x +1 )
                {
                    prev_point->link = lower_run->next;
                    connect_flag = ICV_SINGLE;
                    n++;
                    lower_run = lower_run->next->next;
                }
                else
                {
                    prev_point->link = upper_run;
                    if( upper_run->next->pt.x < lower_run->next->pt.x )
                    {
                        k++;
                        prev_point = upper_run->next;
                        upper_run = upper_run->next->next;
                    }
                    else
                    {
                        connect_flag = ICV_CONNECTING_BELOW;
                        prev_point = lower_run->next;
                        n++;
                        lower_run = lower_run->next->next;
                    }
                }
                break;
            case ICV_CONNECTING_BELOW:
                if( lower_run->pt.x > upper_run->next->pt.x +1 )
                {
                    upper_run->next->link = prev_point;
                    connect_flag = ICV_SINGLE;
                    k++;
                    upper_run = upper_run->next->next;
                }
                else
                {
                    // First point of contour
                    CV_WRITE_SEQ_ELEM( lower_run, writer_int );

                    lower_run->link = prev_point;
                    if( lower_run->next->pt.x < upper_run->next->pt.x )
                    {
                        n++;
                        prev_point = lower_run->next;
                        lower_run = lower_run->next->next;
                    }
                    else
                    {
                        connect_flag = ICV_CONNECTING_ABOVE;
                        k++;
                        prev_point = upper_run->next;
                        upper_run = upper_run->next->next;
                    }
                }
                break;
            }
        }// k, n

        for( ; n < lower_total/2; n++ )
        {
            if( connect_flag != ICV_SINGLE )
            {
                prev_point->link = lower_run->next;
                connect_flag = ICV_SINGLE;
                lower_run = lower_run->next->next;
                continue;
            }
            lower_run->link = lower_run->next;

            //First point of contour
            CV_WRITE_SEQ_ELEM( lower_run, writer_ext );

            lower_run = lower_run->next->next;
        }

        for( ; k < upper_total/2; k++ )
        {
            if( connect_flag != ICV_SINGLE )
            {
                upper_run->next->link = prev_point;
                connect_flag = ICV_SINGLE;
                upper_run = upper_run->next->next;
                continue;
            }
            upper_run->next->link = upper_run;
            upper_run = upper_run->next->next;
        }
        upper_line = lower_line;
        upper_total = lower_total;
    }//i

    upper_run = upper_line;

    //the last line of image
    for( k = 0; k < upper_total/2; k++ )
    {
        upper_run->next->link = upper_run;
        upper_run = upper_run->next->next;
    }

//------//
//------//Find end read contours
    external_contours = cvEndWriteSeq( &writer_ext );
    internal_contours = cvEndWriteSeq( &writer_int );

    for( k = 0; k < 2; k++ )
    {
        CvSeq* contours = k == 0 ? external_contours : internal_contours;

        cvStartReadSeq( contours, &reader );

        for( j = 0; j < contours->total; j++, count++ )
        {
            CvLinkedRunPoint* p_temp;
            CvLinkedRunPoint* p00;
            CvLinkedRunPoint* p01;
            CvSeq* contour;

            CV_READ_SEQ_ELEM( p00, reader );
            p01 = p00;

            if( !p00->link )
                continue;

            cvStartWriteSeq( CV_SEQ_ELTYPE_POINT | CV_SEQ_POLYLINE | CV_SEQ_FLAG_CLOSED,
                             contourHeaderSize, sizeof(CvPoint), storage, &writer );
            do
            {
                CV_WRITE_SEQ_ELEM( p00->pt, writer );
                p_temp = p00;
                p00 = p00->link;
                p_temp->link = 0;
            }
            while( p00 != p01 );

            contour = cvEndWriteSeq( &writer );
            cvBoundingRect( contour, 1 );

            if( k != 0 )
                contour->flags |= CV_SEQ_FLAG_HOLE;

            if( !first )
                prev = first = contour;
            else
            {
                contour->h_prev = prev;
                prev = prev->h_next = contour;
            }
        }
    }

    if( !first )
        count = -1;

    if( result )
        *result = first;

    return count;
}


/****************************************************************************************\
*                         Raster->Chain Tree (Suzuki algorithms)                         *
\****************************************************************************************/

typedef struct _CvContourInfo
{
    int flags;
    struct _CvContourInfo *next;        /* next contour with the same mark value */
    struct _CvContourInfo *parent;      /* information about parent contour */
    CvSeq *contour;             /* corresponding contour (may be 0, if rejected) */
    CvRect rect;                /* bounding rectangle */
    CvPoint origin;             /* origin point (where the contour was traced from) */
    int is_hole;                /* hole flag */
}
_CvContourInfo;


/*
  Structure that is used for sequential retrieving contours from the image.
  It supports both hierarchical and plane variants of Suzuki algorithm.
*/
typedef struct _CvContourScanner
{
    CvMemStorage *storage1;     /* contains fetched contours */
    CvMemStorage *storage2;     /* contains approximated contours
                                   (!=storage1 if approx_method2 != approx_method1) */
    CvMemStorage *cinfo_storage;        /* contains _CvContourInfo nodes */
    CvSet *cinfo_set;           /* set of _CvContourInfo nodes */
    CvMemStoragePos initial_pos;        /* starting storage pos */
    CvMemStoragePos backup_pos; /* beginning of the latest approx. contour */
    CvMemStoragePos backup_pos2;        /* ending of the latest approx. contour */
    schar *img0;                /* image origin */
    schar *img;                 /* current image row */
    int img_step;               /* image step */
    CvSize img_size;            /* ROI size */
    CvPoint offset;             /* ROI offset: coordinates, added to each contour point */
    CvPoint pt;                 /* current scanner position */
    CvPoint lnbd;               /* position of the last met contour */
    int nbd;                    /* current mark val */
    _CvContourInfo *l_cinfo;    /* information about latest approx. contour */
    _CvContourInfo cinfo_temp;  /* temporary var which is used in simple modes */
    _CvContourInfo frame_info;  /* information about frame */
    CvSeq frame;                /* frame itself */
    int approx_method1;         /* approx method when tracing */
    int approx_method2;         /* final approx method */
    int mode;                   /* contour scanning mode:
                                   0 - external only
                                   1 - all the contours w/o any hierarchy
                                   2 - connected components (i.e. two-level structure - external contours and holes),
                                   3 - full hierarchy;
                                   4 - connected components of a multi-level image
                                */
    int subst_flag;
    int seq_type1;              /* type of fetched contours */
    int header_size1;           /* hdr size of fetched contours */
    int elem_size1;             /* elem size of fetched contours */
    int seq_type2;              /*                                       */
    int header_size2;           /*        the same for approx. contours  */
    int elem_size2;             /*                                       */
    _CvContourInfo *cinfo_table[128];
}
_CvContourScanner;



/*
   Initializes scanner structure.
   Prepare image for scanning( clear borders and convert all pixels to 0-1.
*/
static CvContourScanner cvStartFindContours_Impl(void* _img, CvMemStorage* storage, int header_size, int mode, int method, CvPoint offset, int needFillBorder)
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    CvMat stub, *mat = cvGetMat( _img, &stub );

    if( CV_MAT_TYPE(mat->type) == CV_32SC1 && mode == CV_RETR_CCOMP )
        mode = CV_RETR_FLOODFILL;

    if( !((CV_IS_MASK_ARR( mat ) && mode < CV_RETR_FLOODFILL) ||
          (CV_MAT_TYPE(mat->type) == CV_32SC1 && mode == CV_RETR_FLOODFILL)) )
        CV_Error( CV_StsUnsupportedFormat,
                  "[Start]FindContours supports only CV_8UC1 images when mode != CV_RETR_FLOODFILL "
                  "otherwise supports CV_32SC1 images only" );

    CvSize size = cvSize( mat->width, mat->height );
    int step = mat->step;
    uchar* img = (uchar*)(mat->data.ptr);

    if( method < 0 || method > CV_CHAIN_APPROX_TC89_KCOS )
        CV_Error( CV_StsOutOfRange, "" );

    if( header_size < (int) (method == CV_CHAIN_CODE ? sizeof( CvChain ) : sizeof( CvContour )))
        CV_Error( CV_StsBadSize, "" );

    CvContourScanner scanner = (CvContourScanner)cvAlloc( sizeof( *scanner ));
    memset( scanner, 0, sizeof(*scanner) );

    scanner->storage1 = scanner->storage2 = storage;
    scanner->img0 = (schar *) img;
    scanner->img = (schar *) (img + step);
    scanner->img_step = step;
    scanner->img_size.width = size.width - 1;   /* exclude rightest column */
    scanner->img_size.height = size.height - 1; /* exclude bottomost row */
    scanner->mode = mode;
    scanner->offset = offset;
    scanner->pt.x = scanner->pt.y = 1;
    scanner->lnbd.x = 0;
    scanner->lnbd.y = 1;
    scanner->nbd = 2;
    scanner->frame_info.contour = &(scanner->frame);
    scanner->frame_info.is_hole = 1;
    scanner->frame_info.next = 0;
    scanner->frame_info.parent = 0;
    scanner->frame_info.rect = cvRect( 0, 0, size.width, size.height );
    scanner->l_cinfo = 0;
    scanner->subst_flag = 0;
    scanner->frame.flags = CV_SEQ_FLAG_HOLE;
    scanner->approx_method2 = scanner->approx_method1 = method;

    if( method == CV_CHAIN_APPROX_TC89_L1 || method == CV_CHAIN_APPROX_TC89_KCOS )
        scanner->approx_method1 = CV_CHAIN_CODE;

    if( scanner->approx_method1 == CV_CHAIN_CODE )
    {
        scanner->seq_type1 = CV_SEQ_CHAIN_CONTOUR;
        scanner->header_size1 = scanner->approx_method1 == scanner->approx_method2 ? header_size : sizeof( CvChain );
        scanner->elem_size1 = sizeof( char );
    }
    else
    {
        scanner->seq_type1 = CV_SEQ_POLYGON;
        scanner->header_size1 = scanner->approx_method1 == scanner->approx_method2 ? header_size : sizeof( CvContour );
        scanner->elem_size1 = sizeof( CvPoint );
    }

    scanner->header_size2 = header_size;

    if( scanner->approx_method2 == CV_CHAIN_CODE )
    {
        scanner->seq_type2 = scanner->seq_type1;
        scanner->elem_size2 = scanner->elem_size1;
    }
    else
    {
        scanner->seq_type2 = CV_SEQ_POLYGON;
        scanner->elem_size2 = sizeof( CvPoint );
    }

    scanner->seq_type1 = scanner->approx_method1 == CV_CHAIN_CODE ? CV_SEQ_CHAIN_CONTOUR : CV_SEQ_POLYGON;
    scanner->seq_type2 = scanner->approx_method2 == CV_CHAIN_CODE ? CV_SEQ_CHAIN_CONTOUR : CV_SEQ_POLYGON;

    cvSaveMemStoragePos( storage, &(scanner->initial_pos) );

    if( method > CV_CHAIN_APPROX_SIMPLE )
    {
        scanner->storage1 = cvCreateChildMemStorage( scanner->storage2 );
    }

    if( mode > CV_RETR_LIST )
    {
        scanner->cinfo_storage = cvCreateChildMemStorage( scanner->storage2 );
        scanner->cinfo_set = cvCreateSet( 0, sizeof( CvSet ), sizeof( _CvContourInfo ), scanner->cinfo_storage );
    }

    CV_Assert(step >= 0);
    CV_Assert(size.height >= 1);

    /* make zero borders */
    if(needFillBorder)
    {
        int esz = CV_ELEM_SIZE(mat->type);
        memset( img, 0, size.width*esz );
        memset( img + static_cast<size_t>(step) * (size.height - 1), 0, size.width*esz );

        img += step;
        for( int y = 1; y < size.height - 1; y++, img += step )
        {
            for( int k = 0; k < esz; k++ )
                img[k] = img[(size.width - 1)*esz + k] = (schar)0;
        }
    }

    /* converts all pixels to 0 or 1 */
    if( CV_MAT_TYPE(mat->type) != CV_32S )
        cvThreshold( mat, mat, 0, 1, CV_THRESH_BINARY );

    return scanner;
}




CvSeq* TreeToNodeSeq(const void* first, int header_size, CvMemStorage* storage )
{
    CvSeq* allseq = 0;
    CvTreeNodeIterator iterator;

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    allseq = cvCreateSeq( 0, header_size, sizeof(first), storage );

    if( first )
    {
        cvInitTreeNodeIterator( &iterator, first, INT_MAX );

        for(;;)
        {
            void* node = cvNextTreeNode( &iterator );
            if( !node )
                break;
            cvSeqPush( allseq, &node );
        }
    }

    return allseq;
}


/* Copy all sequence elements into single continuous array: */
void* CvtSeqToArray(const CvSeq *seq, void *array, CvSlice slice )
{
    int elem_size, total;
    CvSeqReader reader;
    char *dst = (char*)array;

    if( !seq || !array )
        CV_Error( CV_StsNullPtr, "" );

    elem_size = seq->elem_size;
    total = cvSliceLength( slice, seq )*elem_size;

    if( total == 0 )
        return 0;

    cvStartReadSeq( seq, &reader, 0 );
    cvSetSeqReaderPos( &reader, slice.start_index, 0 );

    do
    {
        int count = (int)(reader.block_max - reader.ptr);
        if( count > total )
            count = total;

        memcpy( dst, reader.ptr, count );
        dst += count;
        reader.block = reader.block->next;
        reader.ptr = reader.block->data;
        reader.block_max = reader.ptr + reader.block->count*elem_size;
        total -= count;
    }
    while( total > 0 );

    return array;
}

static void EndProcessContour( CvContourScanner scanner )
{
    _CvContourInfo *l_cinfo = scanner->l_cinfo;

    if( l_cinfo )
    {
        if( scanner->subst_flag )
        {
            CvMemStoragePos temp;

            cvSaveMemStoragePos( scanner->storage2, &temp );

            if( temp.top == scanner->backup_pos2.top &&
                temp.free_space == scanner->backup_pos2.free_space )
            {
                cvRestoreMemStoragePos( scanner->storage2, &scanner->backup_pos );
            }
            scanner->subst_flag = 0;
        }

        if( l_cinfo->contour )
        {
            cvInsertNodeIntoTree( l_cinfo->contour, l_cinfo->parent->contour,
                                  &(scanner->frame) );
        }
        scanner->l_cinfo = 0;
    }
}

static int TraceContour_32s( int *ptr, int step, int *stop_ptr, int is_hole )
{
    int deltas[16];
    int *i0 = ptr, *i1, *i3, *i4;
    int s, s_end;
    const int   right_flag = INT_MIN;
    const int   new_flag = (int)((unsigned)INT_MIN >> 1);
    const int   value_mask = ~(right_flag | new_flag);
    const int   ccomp_val = *i0 & value_mask;

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    s_end = s = is_hole ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( (*i1 & value_mask) != ccomp_val && s != s_end );

    i3 = i0;

    /* check single pixel domain */
    if( s != s_end )
    {
        /* follow border */
        for( ;; )
        {
            s_end = s;

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( (*i4 & value_mask) == ccomp_val )
                    break;
            }

            if( i3 == stop_ptr || (i4 == i0 && i3 == i1) )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }
    return i3 == stop_ptr;
}

/*
   trace contour until certain point is met.
   returns 1 if met, 0 else.
*/
static int TraceContour( schar *ptr, int step, schar *stop_ptr, int is_hole )
{
    int deltas[16];
    schar *i0 = ptr, *i1, *i3, *i4;
    int s, s_end;

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    assert( (*i0 & -2) != 0 );

    s_end = s = is_hole ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    i3 = i0;

    /* check single pixel domain */
    if( s != s_end )
    {
        /* follow border */
        for( ;; )
        {

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( *i4 != 0 )
                    break;
            }

            if( i3 == stop_ptr || (i4 == i0 && i3 == i1) )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }
    return i3 == stop_ptr;
}



static const CvPoint icvCodeDeltas[8] =
    { CvPoint(1, 0), CvPoint(1, -1), CvPoint(0, -1), CvPoint(-1, -1), CvPoint(-1, 0), CvPoint(-1, 1), CvPoint(0, 1), CvPoint(1, 1) };

/*
    marks domain border with +/-<constant> and stores the contour into CvSeq.
        method:
            <0  - chain
            ==0 - direct
            >0  - simple approximation
*/
static void FetchContour( schar *ptr, int step, CvPoint pt, CvSeq* contour, int _method )
{
    const schar     nbd = 2;
    int             deltas[16];
    CvSeqWriter     writer;
    schar           *i0 = ptr, *i1, *i3, *i4 = 0;
    int             prev_s = -1, s, s_end;
    int             method = _method - 1;

    assert( (unsigned) _method <= CV_CHAIN_APPROX_SIMPLE );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    /* initialize writer */
    cvStartAppendToSeq( contour, &writer );

    if( method < 0 )
        ((CvChain *) contour)->origin = pt;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = (schar) (nbd | -128);
        if( method >= 0 )
        {
            CV_WRITE_SEQ_ELEM( pt, writer );
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            s_end = s;

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( *i4 != 0 )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = (schar) (nbd | -128);
            }
            else if( *i3 == 1 )
            {
                *i3 = nbd;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;

                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else
            {
                if( s != prev_s || method == 0 )
                {
                    CV_WRITE_SEQ_ELEM( pt, writer );
                    prev_s = s;
                }

                pt.x += icvCodeDeltas[s].x;
                pt.y += icvCodeDeltas[s].y;

            }

            if( i4 == i0 && i3 == i1 )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    cvEndWriteSeq( &writer );

    if( _method != CV_CHAIN_CODE )
        cvBoundingRect( contour, 1 );

    assert( (writer.seq->total == 0 && writer.seq->first == 0) ||
            writer.seq->total > writer.seq->first->count ||
            (writer.seq->first->prev == writer.seq->first &&
             writer.seq->first->next == writer.seq->first) );
}

static void
icvFetchContourEx( schar*               ptr,
                   int                  step,
                   CvPoint              pt,
                   CvSeq*               contour,
                   int  _method,
                   int                  nbd,
                   CvRect*              _rect )
{
    int         deltas[16];
    CvSeqWriter writer;
    schar        *i0 = ptr, *i1, *i3, *i4;
    CvRect      rect;
    int         prev_s = -1, s, s_end;
    int         method = _method - 1;

    assert( (unsigned) _method <= CV_CHAIN_APPROX_SIMPLE );
    assert( 1 < nbd && nbd < 128 );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    /* initialize writer */
    cvStartAppendToSeq( contour, &writer );

    if( method < 0 )
        ((CvChain *)contour)->origin = pt;

    rect.x = rect.width = pt.x;
    rect.y = rect.height = pt.y;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = (schar) (nbd | 0x80);
        if( method >= 0 )
        {
            CV_WRITE_SEQ_ELEM( pt, writer );
        }
    }
    else
    {
        i3 = i0;

        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            s_end = s;

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( *i4 != 0 )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = (schar) (nbd | 0x80);
            }
            else if( *i3 == 1 )
            {
                *i3 = (schar) nbd;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;
                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else if( s != prev_s || method == 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
            }

            if( s != prev_s )
            {
                /* update bounds */
                if( pt.x < rect.x )
                    rect.x = pt.x;
                else if( pt.x > rect.width )
                    rect.width = pt.x;

                if( pt.y < rect.y )
                    rect.y = pt.y;
                else if( pt.y > rect.height )
                    rect.height = pt.y;
            }

            prev_s = s;
            pt.x += icvCodeDeltas[s].x;
            pt.y += icvCodeDeltas[s].y;

            if( i4 == i0 && i3 == i1 )  break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;

    cvEndWriteSeq( &writer );

    if( _method != CV_CHAIN_CODE )
        ((CvContour*)contour)->rect = rect;

    assert( (writer.seq->total == 0 && writer.seq->first == 0) ||
            writer.seq->total > writer.seq->first->count ||
            (writer.seq->first->prev == writer.seq->first &&
             writer.seq->first->next == writer.seq->first) );

    if( _rect )  *_rect = rect;
}


static void icvFetchContourEx_32s( int* ptr, int step, CvPoint pt, CvSeq* contour, int _method, CvRect* _rect )
{
    int         deltas[16];
    CvSeqWriter writer;
    int        *i0 = ptr, *i1, *i3, *i4;
    CvRect      rect;
    int         prev_s = -1, s, s_end;
    int         method = _method - 1;
    const int   right_flag = INT_MIN;
    const int   new_flag = (int)((unsigned)INT_MIN >> 1);
    const int   value_mask = ~(right_flag | new_flag);
    const int   ccomp_val = *i0 & value_mask;
    const int   nbd0 = ccomp_val | new_flag;
    const int   nbd1 = nbd0 | right_flag;

    assert( (unsigned) _method <= CV_CHAIN_APPROX_SIMPLE );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    /* initialize writer */
    cvStartAppendToSeq( contour, &writer );

    if( method < 0 )
        ((CvChain *)contour)->origin = pt;

    rect.x = rect.width = pt.x;
    rect.y = rect.height = pt.y;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( (*i1 & value_mask) != ccomp_val && s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = nbd1;
        if( method >= 0 )
        {
            CV_WRITE_SEQ_ELEM( pt, writer );
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            s_end = s;

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( (*i4 & value_mask) == ccomp_val )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = nbd1;
            }
            else if( *i3 == ccomp_val )
            {
                *i3 = nbd0;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;
                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else if( s != prev_s || method == 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
            }

            if( s != prev_s )
            {
                /* update bounds */
                if( pt.x < rect.x )
                    rect.x = pt.x;
                else if( pt.x > rect.width )
                    rect.width = pt.x;

                if( pt.y < rect.y )
                    rect.y = pt.y;
                else if( pt.y > rect.height )
                    rect.height = pt.y;
            }

            prev_s = s;
            pt.x += icvCodeDeltas[s].x;
            pt.y += icvCodeDeltas[s].y;

            if( i4 == i0 && i3 == i1 )  break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;

    cvEndWriteSeq( &writer );

    if( _method != CV_CHAIN_CODE )
        ((CvContour*)contour)->rect = rect;

    assert( (writer.seq->total == 0 && writer.seq->first == 0) ||
           writer.seq->total > writer.seq->first->count ||
           (writer.seq->first->prev == writer.seq->first &&
            writer.seq->first->next == writer.seq->first) );

    if( _rect ) 
        *_rect = rect;
}


typedef struct _CvPtInfo
{
    CvPoint pt;
    int k;                      /* support region */
    int s;                      /* curvature value */
    struct _CvPtInfo *next;
}
_CvPtInfo;

/* curvature: 0 - 1-curvature, 1 - k-cosine curvature. */
CvSeq* icvApproximateChainTC89( CvChain* chain, int header_size, CvMemStorage* storage, int method )
{
    static const int abs_diff[] = { 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1 };

    cv::AutoBuffer<_CvPtInfo> buf(chain->total + 8);

    _CvPtInfo       temp;
    _CvPtInfo       *array = buf, *first = 0, *current = 0, *prev_current = 0;
    int             i, j, i1, i2, s, len;
    int             count = chain->total;

    CvChainPtReader reader;
    CvSeqWriter     writer;
    CvPoint         pt = chain->origin;

    CV_Assert( CV_IS_SEQ_CHAIN_CONTOUR( chain ));
    CV_Assert( header_size >= (int)sizeof(CvContour) );

    cvStartWriteSeq( (chain->flags & ~CV_SEQ_ELTYPE_MASK) | CV_SEQ_ELTYPE_POINT,
                     header_size, sizeof( CvPoint ), storage, &writer );

    if( chain->total == 0 )
    {
        CV_WRITE_SEQ_ELEM( pt, writer );
        return cvEndWriteSeq( &writer );
    }

    reader.code = 0;
    cvStartReadChainPoints( chain, &reader );

    temp.next = 0;
    current = &temp;

    /* Pass 0.
       Restores all the digital curve points from the chain code.
       Removes the points (from the resultant polygon)
       that have zero 1-curvature */
    for( i = 0; i < count; i++ )
    {
        int prev_code = *reader.prev_elem;

        reader.prev_elem = reader.ptr;
        CV_READ_CHAIN_POINT( pt, reader );

        /* calc 1-curvature */
        s = abs_diff[reader.code - prev_code + 7];

        if( method <= CV_CHAIN_APPROX_SIMPLE )
        {
            if( method == CV_CHAIN_APPROX_NONE || s != 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
            }
        }
        else
        {
            if( s != 0 )
                current = current->next = array + i;
            array[i].s = s;
            array[i].pt = pt;
        }
    }

    //assert( pt.x == chain->origin.x && pt.y == chain->origin.y );

    if( method <= CV_CHAIN_APPROX_SIMPLE )
        return cvEndWriteSeq( &writer );

    current->next = 0;

    len = i;
    current = temp.next;

    assert( current );

    /* Pass 1.
       Determines support region for all the remained points */
    do
    {
        CvPoint pt0;
        int k, l = 0, d_num = 0;

        i = (int)(current - array);
        pt0 = array[i].pt;

        /* determine support region */
        for( k = 1;; k++ )
        {
            int lk, dk_num;
            int dx, dy;
            Cv32suf d;

            assert( k <= len );

            /* calc indices */
            i1 = i - k;
            i1 += i1 < 0 ? len : 0;
            i2 = i + k;
            i2 -= i2 >= len ? len : 0;

            dx = array[i2].pt.x - array[i1].pt.x;
            dy = array[i2].pt.y - array[i1].pt.y;

            /* distance between p_(i - k) and p_(i + k) */
            lk = dx * dx + dy * dy;

            /* distance between p_i and the line (p_(i-k), p_(i+k)) */
            dk_num = (pt0.x - array[i1].pt.x) * dy - (pt0.y - array[i1].pt.y) * dx;
            d.f = (float) (((double) d_num) * lk - ((double) dk_num) * l);

            if( k > 1 && (l >= lk || ((d_num > 0 && d.i <= 0) || (d_num < 0 && d.i >= 0))))
                break;

            d_num = dk_num;
            l = lk;
        }

        current->k = --k;

        /* determine cosine curvature if it should be used */
        if( method == CV_CHAIN_APPROX_TC89_KCOS )
        {
            /* calc k-cosine curvature */
            for( j = k, s = 0; j > 0; j-- )
            {
                double temp_num;
                int dx1, dy1, dx2, dy2;
                Cv32suf sk;

                i1 = i - j;
                i1 += i1 < 0 ? len : 0;
                i2 = i + j;
                i2 -= i2 >= len ? len : 0;

                dx1 = array[i1].pt.x - pt0.x;
                dy1 = array[i1].pt.y - pt0.y;
                dx2 = array[i2].pt.x - pt0.x;
                dy2 = array[i2].pt.y - pt0.y;

                if( (dx1 | dy1) == 0 || (dx2 | dy2) == 0 )
                    break;

                temp_num = dx1 * dx2 + dy1 * dy2;
                temp_num =
                    (float) (temp_num /
                             sqrt( ((double)dx1 * dx1 + (double)dy1 * dy1) *
                                   ((double)dx2 * dx2 + (double)dy2 * dy2) ));
                sk.f = (float) (temp_num + 1.1);

                assert( 0 <= sk.f && sk.f <= 2.2 );
                if( j < k && sk.i <= s )
                    break;

                s = sk.i;
            }
            current->s = s;
        }
        current = current->next;
    }
    while( current != 0 );

    prev_current = &temp;
    current = temp.next;

    /* Pass 2.
       Performs non-maxima suppression */
    do
    {
        int k2 = current->k >> 1;

        s = current->s;
        i = (int)(current - array);

        for( j = 1; j <= k2; j++ )
        {
            i2 = i - j;
            i2 += i2 < 0 ? len : 0;

            if( array[i2].s > s )
                break;

            i2 = i + j;
            i2 -= i2 >= len ? len : 0;

            if( array[i2].s > s )
                break;
        }

        if( j <= k2 )           /* exclude point */
        {
            prev_current->next = current->next;
            current->s = 0;     /* "clear" point */
        }
        else
            prev_current = current;
        current = current->next;
    }
    while( current != 0 );

    /* Pass 3.
       Removes non-dominant points with 1-length support region */
    current = temp.next;
    assert( current );
    prev_current = &temp;

    do
    {
        if( current->k == 1 )
        {
            s = current->s;
            i = (int)(current - array);

            i1 = i - 1;
            i1 += i1 < 0 ? len : 0;

            i2 = i + 1;
            i2 -= i2 >= len ? len : 0;

            if( s <= array[i1].s || s <= array[i2].s )
            {
                prev_current->next = current->next;
                current->s = 0;
            }
            else
                prev_current = current;
        }
        else
            prev_current = current;
        current = current->next;
    }
    while( current != 0 );

    if( method == CV_CHAIN_APPROX_TC89_KCOS )
        goto copy_vect;

    /* Pass 4.
       Cleans remained couples of points */
    assert( temp.next );

    if( array[0].s != 0 && array[len - 1].s != 0 )      /* specific case */
    {
        for( i1 = 1; i1 < len && array[i1].s != 0; i1++ )
        {
            array[i1 - 1].s = 0;
        }
        if( i1 == len )
            goto copy_vect;     /* all points survived */
        i1--;

        for( i2 = len - 2; i2 > 0 && array[i2].s != 0; i2-- )
        {
            array[i2].next = 0;
            array[i2 + 1].s = 0;
        }
        i2++;

        if( i1 == 0 && i2 == len - 1 )  /* only two points */
        {
            i1 = (int)(array[0].next - array);
            array[len] = array[0];      /* move to the end */
            array[len].next = 0;
            array[len - 1].next = array + len;
        }
        temp.next = array + i1;
    }

    current = temp.next;
    first = prev_current = &temp;
    count = 1;

    /* do last pass */
    do
    {
        if( current->next == 0 || current->next - current != 1 )
        {
            if( count >= 2 )
            {
                if( count == 2 )
                {
                    int s1 = prev_current->s;
                    int s2 = current->s;

                    if( s1 > s2 || (s1 == s2 && prev_current->k <= current->k) )
                        /* remove second */
                        prev_current->next = current->next;
                    else
                        /* remove first */
                        first->next = current;
                }
                else
                    first->next->next = current;
            }
            first = current;
            count = 1;
        }
        else
            count++;
        prev_current = current;
        current = current->next;
    }
    while( current != 0 );

copy_vect:

    // gather points
    current = temp.next;
    assert( current );

    do
    {
        CV_WRITE_SEQ_ELEM( current->pt, writer );
        current = current->next;
    }
    while( current != 0 );

    return cvEndWriteSeq( &writer );
}


CvSeq* FindNextContour( CvContourScanner scanner )
{
    if( !scanner )
        CV_Error( CV_StsNullPtr, "" );

#if CV_SSE2
    bool haveSIMD = cv::checkHardwareSupport(CPU_SSE2);
#endif

    CV_Assert(scanner->img_step >= 0);

    EndProcessContour( scanner );

    /* initialize local state */
    schar* img0 = scanner->img0;
    schar* img = scanner->img;
    int step = scanner->img_step;
    int step_i = step / sizeof(int);
    int x = scanner->pt.x;
    int y = scanner->pt.y;
    int width = scanner->img_size.width;
    int height = scanner->img_size.height;
    int mode = scanner->mode;
    CvPoint lnbd = scanner->lnbd;
    int nbd = scanner->nbd;
    int prev = img[x - 1];
    int new_mask = -2;

    if( mode == CV_RETR_FLOODFILL )
    {
        prev = ((int*)img)[x - 1];
        new_mask = INT_MIN / 2;
    }

    for( ; y < height; y++, img += step )
    {
        int* img0_i = 0;
        int* img_i = 0;
        int p = 0;

        if( mode == CV_RETR_FLOODFILL )
        {
            img0_i = (int*)img0;
            img_i = (int*)img;
        }

        for( ; x < width; x++ )
        {
            if( img_i )
            {
                for( ; x < width && ((p = img_i[x]) == prev || (p & ~new_mask) == (prev & ~new_mask)); x++ )
                    prev = p;
            }
            else
            {
#if CV_SSE2
                if ((p = img[x]) != prev) {
                    goto _next_contour;
                } else if (haveSIMD) {

                    __m128i v_prev = _mm_set1_epi8((char)prev);
                    int v_size = width - 32;

                    for (; x <= v_size; x += 32) {
                        __m128i v_p1 = _mm_loadu_si128((const __m128i*)(img + x));
                        __m128i v_p2 = _mm_loadu_si128((const __m128i*)(img + x + 16));

                        __m128i v_cmp1 = _mm_cmpeq_epi8(v_p1, v_prev);
                        __m128i v_cmp2 = _mm_cmpeq_epi8(v_p2, v_prev);

                        unsigned int mask1 = _mm_movemask_epi8(v_cmp1);
                        unsigned int mask2 = _mm_movemask_epi8(v_cmp2);

                        mask1 ^= 0x0000ffff;
                        mask2 ^= 0x0000ffff;

                        if (mask1) {
                            p = img[(x += trailingZeros(mask1))];
                            goto _next_contour;
                        }

                        if (mask2) {
                            p = img[(x += trailingZeros(mask2 << 16))];
                            goto _next_contour;
                        }
                    }

                    if(x <= width - 16) {
                        __m128i v_p = _mm_loadu_si128((__m128i*)(img + x));

                        unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v_p, v_prev)) ^ 0x0000ffff;

                        if (mask) {
                            p = img[(x += trailingZeros(mask))];
                            goto _next_contour;
                        }
                        x += 16;
                    }
                }
#endif
                for( ; x < width && (p = img[x]) == prev; x++ )
                    ;
            }

            if( x >= width )
                break;
#if CV_SSE2
        _next_contour:
#endif
            {
                _CvContourInfo *par_info = 0;
                _CvContourInfo *l_cinfo = 0;
                CvSeq *seq = 0;
                int is_hole = 0;
                CvPoint origin;

                /* if not external contour */
                if( (!img_i && !(prev == 0 && p == 1)) ||
                    (img_i && !(((prev & new_mask) != 0 || prev == 0) && (p & new_mask) == 0)) )
                {
                    /* check hole */
                    if( (!img_i && (p != 0 || prev < 1)) ||
                        (img_i && ((prev & new_mask) != 0 || (p & new_mask) != 0)))
                        goto resume_scan;

                    if( prev & new_mask )
                    {
                        lnbd.x = x - 1;
                    }
                    is_hole = 1;
                }

                if( mode == 0 && (is_hole || img0[lnbd.y * static_cast<size_t>(step) + lnbd.x] > 0) )
                    goto resume_scan;

                origin.y = y;
                origin.x = x - is_hole;

                /* find contour parent */
                if( mode <= 1 || (!is_hole && (mode == CV_RETR_CCOMP || mode == CV_RETR_FLOODFILL)) || lnbd.x <= 0 )
                {
                    par_info = &(scanner->frame_info);
                }
                else
                {
                    int lval = (img0_i ?
                        img0_i[lnbd.y * static_cast<size_t>(step_i) + lnbd.x] :
                        (int)img0[lnbd.y * static_cast<size_t>(step) + lnbd.x]) & 0x7f;
                    _CvContourInfo *cur = scanner->cinfo_table[lval];

                    /* find the first bounding contour */
                    while( cur )
                    {
                        if( (unsigned) (lnbd.x - cur->rect.x) < (unsigned) cur->rect.width &&
                            (unsigned) (lnbd.y - cur->rect.y) < (unsigned) cur->rect.height )
                        {
                            if( par_info )
                            {
                                if( (img0_i && TraceContour_32s( img0_i + par_info->origin.y * static_cast<size_t>(step_i) +
                                                          par_info->origin.x, step_i, img_i + lnbd.x, par_info->is_hole ) > 0) ||
                                    (!img0_i && TraceContour( img0 + par_info->origin.y * static_cast<size_t>(step) +
                                                      par_info->origin.x, step, img + lnbd.x, par_info->is_hole ) > 0) )
                                    break;
                            }
                            par_info = cur;
                        }
                        cur = cur->next;
                    }

                    assert( par_info != 0 );

                    /* if current contour is a hole and previous contour is a hole or
                       current contour is external and previous contour is external then
                       the parent of the contour is the parent of the previous contour else
                       the parent is the previous contour itself. */
                    if( par_info->is_hole == is_hole )
                    {
                        par_info = par_info->parent;
                        /* every contour must have a parent
                           (at least, the frame of the image) */
                        if( !par_info )
                            par_info = &(scanner->frame_info);
                    }

                    /* hole flag of the parent must differ from the flag of the contour */
                    assert( par_info->is_hole != is_hole );
                    if( par_info->contour == 0 )        /* removed contour */
                        goto resume_scan;
                }

                lnbd.x = x - is_hole;

                cvSaveMemStoragePos( scanner->storage2, &(scanner->backup_pos) );

                seq = cvCreateSeq( scanner->seq_type1, scanner->header_size1,
                                   scanner->elem_size1, scanner->storage1 );
                seq->flags |= is_hole ? CV_SEQ_FLAG_HOLE : 0;

                /* initialize header */
                if( mode <= 1 )
                {
                    l_cinfo = &(scanner->cinfo_temp);
                    FetchContour( img + x - is_hole, step, cvPoint( origin.x + scanner->offset.x, origin.y + scanner->offset.y), seq, scanner->approx_method1 );
                }
                else
                {
                    union { _CvContourInfo* ci; CvSetElem* se; } v;
                    v.ci = l_cinfo;
                    cvSetAdd( scanner->cinfo_set, 0, &v.se );
                    l_cinfo = v.ci;
                    int lval;

                    if( img_i )
                    {
                        lval = img_i[x - is_hole] & 127;
                        icvFetchContourEx_32s(img_i + x - is_hole, step_i,
                                              cvPoint( origin.x + scanner->offset.x,
                                                       origin.y + scanner->offset.y),
                                              seq, scanner->approx_method1,
                                              &(l_cinfo->rect) );
                    }
                    else
                    {
                        lval = nbd;
                        // change nbd
                        nbd = (nbd + 1) & 127;
                        nbd += nbd == 0 ? 3 : 0;
                        icvFetchContourEx( img + x - is_hole, step,
                                           cvPoint( origin.x + scanner->offset.x,
                                                    origin.y + scanner->offset.y),
                                           seq, scanner->approx_method1,
                                           lval, &(l_cinfo->rect) );
                    }
                    l_cinfo->rect.x -= scanner->offset.x;
                    l_cinfo->rect.y -= scanner->offset.y;

                    l_cinfo->next = scanner->cinfo_table[lval];
                    scanner->cinfo_table[lval] = l_cinfo;
                }

                l_cinfo->is_hole = is_hole;
                l_cinfo->contour = seq;
                l_cinfo->origin = origin;
                l_cinfo->parent = par_info;

                if( scanner->approx_method1 != scanner->approx_method2 )
                {
                    l_cinfo->contour = icvApproximateChainTC89( (CvChain *) seq,
                                                      scanner->header_size2,
                                                      scanner->storage2,
                                                      scanner->approx_method2 );
                    cvClearMemStorage( scanner->storage1 );
                }

                l_cinfo->contour->v_prev = l_cinfo->parent->contour;

                if( par_info->contour == 0 )
                {
                    l_cinfo->contour = 0;
                    if( scanner->storage1 == scanner->storage2 )
                    {
                        cvRestoreMemStoragePos( scanner->storage1, &(scanner->backup_pos) );
                    }
                    else
                    {
                        cvClearMemStorage( scanner->storage1 );
                    }
                    p = img[x];
                    goto resume_scan;
                }

                cvSaveMemStoragePos( scanner->storage2, &(scanner->backup_pos2) );
                scanner->l_cinfo = l_cinfo;
                scanner->pt.x = !img_i ? x + 1 : x + 1 - is_hole;
                scanner->pt.y = y;
                scanner->lnbd = lnbd;
                scanner->img = (schar *) img;
                scanner->nbd = nbd;
                return l_cinfo->contour;

            resume_scan:

                prev = p;
                /* update lnbd */
                if( prev & -2 )
                {
                    lnbd.x = x;
                }
            }                   /* end of prev != p */
        }                       /* end of loop on x */

        lnbd.x = 0;
        lnbd.y = y + 1;
        x = 1;
        prev = 0;
    }                           /* end of loop on y */

    return 0;
}



static int FindContours_Impl(void* img, CvMemStorage* storage, CvSeq** firstContour, int cntHeaderSize, int mode, int method, CvPoint offset, int needFillBorder )
{
    CvContourScanner scanner = 0;
    CvSeq *contour = 0;
    int count = -1;

    if( !firstContour )
        CV_Error(CV_StsNullPtr, "NULL double CvSeq pointer" );

    *firstContour = 0;

    if( method == CV_LINK_RUNS )
    {
        if( offset.x != 0 || offset.y != 0 )
            CV_Error( CV_StsOutOfRange, "Nonzero offset is not supported in CV_LINK_RUNS yet" );

        count = icvFindContoursInInterval( img, storage, firstContour, cntHeaderSize );
    }
    else
    {
        try
        {
            scanner = cvStartFindContours_Impl( img, storage, cntHeaderSize, mode, method, offset, needFillBorder);

            do
            {
                count++;
                contour = FindNextContour( scanner );
            }
            while( contour != 0 );
        }
        catch(...)
        {
            if( scanner )
                cvEndFindContours(&scanner);
            throw;
        }

        *firstContour = cvEndFindContours( &scanner );
    }

    return count;
}



void findContours(cv::Mat _image, std::vector< std::vector<cv::Point> >& _contours, std::vector<cv::Vec4i>& _hierarchy, int mode, int method)
{
    // output must be of type vector<vector<Point>>
    cv::Mat image;
    copyMakeBorder(_image, image, 1, 1, 1, 1, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0) );
    
    cv::MemStorage storage(cvCreateMemStorage());
    
    CvMat _cimage = image;
    CvSeq* _ccontours = 0;
    
    FindContours_Impl(&_cimage, storage, &_ccontours, sizeof(CvContour), mode, method, cv::Point(-1, -1), 0);
    
    cv::Seq<CvSeq*> all_contours( TreeToNodeSeq( _ccontours, sizeof(CvSeq), storage ));
    
    int total = (int)all_contours.size();

    cv::SeqIterator<CvSeq*> it = all_contours.begin();
    
    for(int i = 0; i < total; i++, ++it)
    {
        CvSeq* c = *it;
        ((CvContour*)c)->color = i;

        //_contours.create((int)c->total, 1, CV_32SC2, i, true);
        
        //cv::Mat ci = _contours.getMat(i);
        //CV_Assert( ci.isContinuous() );
        //CvtSeqToArray(c, ci.ptr(), CV_WHOLE_SEQ);
    }


    it = all_contours.begin();
    for(int i = 0; i < total; i++, ++it)
    {
        CvSeq* c = *it;
        int h_next = c->h_next ? ((CvContour*)c->h_next)->color : -1;
        int h_prev = c->h_prev ? ((CvContour*)c->h_prev)->color : -1;
        int v_next = c->v_next ? ((CvContour*)c->v_next)->color : -1;
        int v_prev = c->v_prev ? ((CvContour*)c->v_prev)->color : -1;
        
        _hierarchy[i] = cv::Vec4i(h_next, h_prev, v_next, v_prev);
    }
}


int checkChessboardBinary(const cv::Mat & img, const cv::Size & size)
{
    CV_Assert(img.channels() == 1 && img.depth() == CV_8U);
    #ifdef SAVE_TEMP_RES
        imwrite("img_before_morphology.png", img);
    #endif

    cv::Mat black = img.clone();

    for(int i = 0; i <= 3; i++)
    {
        if( 0 != i ) // first iteration keeps original images
        {
            //erode(white, white, cv::Mat(), cv::Point(-1, -1), 1);
            dilate(black, black, cv::Mat(), cv::Point(-1, -1), 1);

            #ifdef SAVE_TEMP_RES
                std::ostringstream str;
                str << i;
                imwrite("dilate_black" + str.str() + ".png", black);
            #endif
        }
        
        std::vector<float> quads;
        std::vector< std::vector<cv::Point> > contours;
        std::vector< cv::Vec4i > hierarchy;

        findContours(black, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        GetBlackQuadHypotheses(contours, hierarchy, quads);
        if( checkBlackQuads(quads, size) )
            return 1;
    }
    return 0;
}


int main( int argc, char** argv )
{
    // ./rotatedRect threshold_mean.png
    cv::Mat thresh_img_new = cv::imread(argv[1], 1);
    if (thresh_img_new.channels() != 1)
    {
        cvtColor(thresh_img_new, thresh_img_new, cv::COLOR_BGR2GRAY);
    }
    g_img = thresh_img_new.clone();
    cv::Size cornerSize(8, 6);

    if(checkChessboardBinary(thresh_img_new, cornerSize) <= 0) //fall back to the old method
    {
        printf("fall back to the old method.\n");
        return -1;
    }

    imshow("rectangles", thresh_img_new);
    cv::waitKey(0);

    return 0;
}
