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


int checkVector2(cv::Mat &mat, int _elemChannels, int _depth, bool _requireContinuous)
{
    if( (mat.depth() == _depth || _depth <= 0) && (mat.isContinuous() || !_requireContinuous) )
    {
        if(mat.dims == 2)
        {
            if( (( mat.rows == 1 || mat.cols == 1) && mat.channels() == _elemChannels) || (mat.cols == _elemChannels && mat.channels() == 1))
            {
                return (int)(mat.total() * mat.channels()/_elemChannels);
            }
        }
        else if( mat.dims == 3 )
        {
            if( mat.channels() == 1 && mat.size.p[2] == _elemChannels && ( mat.size.p[0] == 1 || mat.size.p[1] == 1)
                && ( mat.isContinuous() || mat.step.p[1] == mat.step.p[2] * mat.size.p[2] ) )
            {
                return (int)(mat.total() * mat.channels()/_elemChannels);
            }
        }
    }
    return -1;
}

/* Parameters:
	 points      - convex hull vertices ( any orientation )
	 n           - number of vertices
	 mode        - concrete application of algorithm
	               can be  CV_CALIPERS_MAXDIST   or
	               CV_CALIPERS_MINAREARECT
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

void rotatingCalipers( std::vector< cv::Point > &points, int n, int mode, float* out )
{
    float minarea = FLT_MAX;
    float max_dist = 0;
    char buffer[32] = {};
    int i, k;
    cv::AutoBuffer<float> abuf(n*3);
    float* inv_vect_length = abuf;
    cv::Point2f* vect = (cv::Point2f*)(inv_vect_length + n);
    int left = 0, bottom = 0, right = 0, top = 0;
    int seq[4] = { -1, -1, -1, -1 };

    /* rotating calipers sides will always have coordinates
     (a,b) (-b,a) (-a,-b) (b, -a)
     */
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
    for( k = 0; k < n; k++ )
    {
        /* sinus of minimal angle */
        /*float sinus;*/

        /* compute cosine of angle between calipers side and polygon edge */
        /* dp - dot product */
        float dp[4] = {
            +base_a * vect[seq[0]].x + base_b * vect[seq[0]].y,
            -base_b * vect[seq[1]].x + base_a * vect[seq[1]].y,
            -base_a * vect[seq[2]].x - base_b * vect[seq[2]].y,
            +base_b * vect[seq[3]].x - base_a * vect[seq[3]].y,
        };

        float maxcos = dp[0] * inv_vect_length[seq[0]];

        /* number of calipers edges, that has minimal angle with edge */
        int main_element = 0;

        /* choose minimal angle */
        for (i = 1; i < 4; ++i )
        {
            float cosalpha = dp[i] * inv_vect_length[seq[i]];
            if (cosalpha > maxcos)
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

        switch (mode)
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

    switch (mode)
    {
        case CALIPERS_MINAREARECT:
        {
            float *buf = (float *) buffer;

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


cv::RotatedRect minRectArea(std::vector<cv::Point> _points )
{
    //cv::Mat hull;
    std::vector<cv::Point> hpoints;

    cv::Point2f out[3];
    cv::RotatedRect box;

    convexHull(_points, hpoints, true, true);
    int n = (int)hpoints.size();

    /*  
    if( hull.depth() != CV_32F ){
        //cv::Mat temp;
        hull.convertTo(hull, CV_32F);
        //hull = temp;
    }
    */
    
    //int n = checkVector(hull, 2, -1, true);
    //std::cout <<" n: " << n << std::endl;
    
    //const cv::Point* hpoints = hull.ptr<cv::Point>();

    if( n > 2 )
    {
        rotatingCalipers( hpoints, n, CALIPERS_MINAREARECT, (float*)out );
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


void drawBox(cv::RotatedRect box)
{
    cv::Mat image(553, 478, CV_8UC3, cv::Scalar(0));
    cv::Point2f vertices[4];
    box.points(vertices);
    
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));

    imshow("rectangles", image);
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
        if(pt->size() < 40)
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


void GetQuadrangleHypotheses(const std::vector<std::vector< cv::Point > > & contours, 
        const std::vector< cv::Vec4i > & hierarchy, std::vector<std::pair<float, int> >& quads, int class_id)
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
        if(hierarchy[idx][3] != -1 || hierarchy[idx][2] != -1)
            continue; // skip holes
        
        // too small, min_box_size * 4
        if(pt->size() < 40)
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
        quads.push_back( std::pair<float, int>(box_size, class_id) );
    }
    std::cout << "quads.size(): " << quads.size() << std::endl;
}


inline bool less_pred(const std::pair<float, int>& p1, const std::pair<float, int>& p2)
{
    return p1.first < p2.first;
}


bool checkBlackQuads(std::vector<float> & quads, const cv::Size & size)
{
    const unsigned int black_quads_count = (size.width+1) * (size.height+1) / 2 + 1;
    const unsigned int min_quads_count = black_quads_count / 2;
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


static bool checkQuads(vector<pair<float, int> > & quads, const cv::Size & size)
{
    const unsigned int min_quads_count = size.width*size.height/2;
    const float size_rel_para = 1.4f;

    // both black and write half of the total number of quads
    //std::cout << "min_quads_count:" << min_quads_count << std::endl;
    std::sort(quads.begin(), quads.end(), less_pred);

    // now check if there are many hypotheses with similar sizes
    // do this by floodfill-style algorithm
    for(unsigned int i = 0; i < quads.size(); i++)
    {
        unsigned int j = i + 1;
        for(; j < quads.size(); j++)
        {
            if(quads[j].first/quads[i].first > size_rel_para)
            {
                break;
            }
        }

        if(j - i + 1 > min_quads_count )
        {
            // check the number of black and white squares
            std::vector<int> counts(2, 0);
            //countClasses(quads, i, j, counts);

            for(unsigned int n = i; n != j; n++)
            {
                counts[quads[n].second]++;
            }

            //const int white_count = cvRound( floor(size.width/2.0) * floor(size.height/2.0) );
            if(counts[0] < min_quads_count / 2 || counts[1] < min_quads_count / 2)
                continue;

            return true;
        }
    }
    return false;
}

void bimInverse(cv::Mat & img)
{
    for (int j = 0; j < img.rows; j++)
    {
        unsigned char * row = img.ptr(j);
        for (int i = 0; i < img.cols; i++)
            row[i] = ~row[i];
    }
}

int checkChessboardBinary(const cv::Mat & img, const cv::Size & size)
{
    CV_Assert(img.channels() == 1 && img.depth() == CV_8U);
    #ifdef SAVE_TEMP_RES
        imwrite("img_before_morphology.png", img);
    #endif

    cv::Mat white = img.clone();
    cv::Mat black = img.clone();

    for(int i = 0; i <= 3; i++)
    {
        if( 0 != i ) // first iteration keeps original images
        {
            erode(white, white, cv::Mat(), cv::Point(-1, -1), 1);
            dilate(black, black, cv::Mat(), cv::Point(-1, -1), 1);

            #ifdef SAVE_TEMP_RES
                std::ostringstream str;
                str << i;
                imwrite("erode_white" + str.str() + ".png", white);
                imwrite("dilate_black.png" + str.str() + ".png", black);
            #endif
        }
        
        vector<pair<float, int> > quads;
    
        std::vector< std::vector<cv::Point> > contours;
        std::vector< cv::Vec4i > hierarchy;

        findContours(white, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        GetQuadrangleHypotheses(contours, hierarchy, quads, 1);
        
        contours.clear();
        hierarchy.clear();
        
        cv::Mat thresh = black.clone();
        
        bimInverse(thresh);

        findContours(thresh, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        GetQuadrangleHypotheses(contours, hierarchy, quads, 0);
    
        if ( checkQuads(quads, size) )
            return 1;
    }
    return 0;
}


int checkChessboardBinary2(const cv::Mat & img, const cv::Size & size)
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
                imwrite("dilate_black.png" + str.str() + ".png", black);
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
/*  cv::Mat image(200, 200, CV_8UC3, cv::Scalar(0));
    cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(100,100), cv::Size2f(100,50), 30);
    
    cv::Point2f vertices[4];
    rRect.points(vertices);
    
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));
    cv::Rect brect = rRect.boundingRect();
    rectangle(image, brect, cv::Scalar(255,0,0));
    imshow("rectangles", image);
    cv::waitKey(0);
*/
    // ./rotatedRect threshold_mean.png
    cv::Mat thresh_img_new = cv::imread(argv[1], 1);
    if (thresh_img_new.channels() != 1)
    {
        cvtColor(thresh_img_new, thresh_img_new, cv::COLOR_BGR2GRAY);
    }
    g_img = thresh_img_new.clone();
    cv::Size cornerSize(8, 6);

    if(checkChessboardBinary2(thresh_img_new, cornerSize) <= 0) //fall back to the old method
    {
        printf("fall back to the old method.\n");
        return -1;
    }

    imshow("rectangles", thresh_img_new);
    cv::waitKey(0);

    return 0;
}
