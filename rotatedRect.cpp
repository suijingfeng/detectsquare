#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#define SAVE_TEMP_RES   1

#include "convexHull.h"
#include "rotatingCalipers.h"
#include "findContours.h"
#include "threshold.h"

cv::Mat g_img;


void drawBox(cv::Mat &img, cv::RotatedRect box)
{
    if (img.channels() == 1)
        cvtColor(img, img, cv::COLOR_GRAY2BGR);
    
    cv::Point2f vertices[4];
    box.points(vertices);
    
    //for (int i = 0; i < 4; i++)
    //    line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));
    if ( !img.empty() )
    {    
        line(img, vertices[0], vertices[1], cv::Scalar(0,255,0), 2);
        line(img, vertices[1], vertices[2], cv::Scalar(0,255,0), 2);
        line(img, vertices[2], vertices[3], cv::Scalar(0,255,0), 2);
        line(img, vertices[3], vertices[0], cv::Scalar(0,255,0), 2);

        imshow("rectangles", img);
        cv::waitKey(0);
    }
}


void GetBlackQuadHypotheses(const std::vector<std::vector< cv::Point > > &contours, const std::vector< cv::Vec4i > &hierarchy, std::vector<float> &quads)
{
    const float min_aspect_ratio = 0.5f, max_aspect_ratio = 2.0f;
    const float min_quad_size = 10.0f, max_quad_pixel = 250;
    
    std::vector< std::vector< cv::Point > >::const_iterator pt;
    printf("contours.size(): %ld\n", contours.size());

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
        printf("pt->size(): %ld\n", pt->size() );
        
        // cv::RotatedRect box = minRectArea(*pt);
        //convexHull(*pt, hpoints, false, true);

        std::vector<cv::Point> hpoints = convex_hull(*pt);
    
        cv::RotatedRect box = minAreaRect(hpoints);

        float box_size = MAX(box.size.width, box.size.height);
        
        if(box_size < min_quad_size || box_size > max_quad_pixel)
            continue;

        float aspect_ratio = box.size.width / MAX(box.size.height, 1);
        if(aspect_ratio < min_aspect_ratio || aspect_ratio > max_aspect_ratio)
            continue;
        
        drawBox(g_img, box);
        quads.push_back( box_size );
    }
    printf("quads.size(): %ld\n", quads.size() );
}


bool checkBlackQuads(std::vector<float> &quads, const cv::Size &size)
{
    const unsigned int black_quads_count = (size.width+1) * (size.height+1) / 2 + 1;
    const unsigned int min_quads_count = black_quads_count * 0.75;
    const float quad_aspect_ratio = 1.4f;

    std::sort(quads.begin(), quads.end());
    
    // now check if there are many hypotheses with similar sizes
    for(unsigned int i = 0; i < quads.size(); i++)
    {
        unsigned int j = i + 1;
        for( ; j < quads.size(); j++)
            if(quads[j]/quads[i] > quad_aspect_ratio)
                break;

        if(j - i + 1 > min_quads_count )
            return true; // check the number of black squares
    }
    return false;
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

        findContours(black, contours, hierarchy);
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
