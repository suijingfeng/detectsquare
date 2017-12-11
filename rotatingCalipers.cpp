#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

/* Parameters:
    points: convex hull vertices (any orientation)
    we will use usual cartesian coordinates 
*/

cv::RotatedRect rotatingCalipersMinAreaRect(std::vector<cv::Point> &points)
{
    cv::RotatedRect box;
    int n = points.size();

    if(n > 2)
    {
        std::vector<float> vect_length(n);
        std::vector<cv::Point> vect(n);
        
        int seq[4] = {0, 0, 0, 0};
        float minarea;
        
        // Base vector (base_a, base_b) should be initialized by (-1,0), if convex hull orientation is anti-clockwised
        float base_a = 1.0, base_b = 0;
        // rotating calipers sides will always have coordinates (a,b) (-b,a) (-a,-b) (b, -a)

        { 
            float left_x = points[0].x, right_x = points[0].x;
            float top_y = points[0].y, bottom_y = points[0].y;

            for(int i = 0; i < n; i++)
            {
                if( points[i].x < left_x )
                {
                    left_x = points[i].x;
                    seq[3] = i; // left
                }
                else if( points[i].x > right_x )
                {
                    right_x = points[i].x;
                    seq[1] = i; // right
                }

                if( points[i].y > top_y )
                {
                    top_y = points[i].y;
                    seq[2] = i; // top
                }
                else if( points[i].y < bottom_y )
                {
                    bottom_y = points[i].y;
                    seq[0] = i; // bottom
                }
                vect[i].x = points[(i+1) % n].x - points[i].x;
                vect[i].y = points[(i+1) % n].y - points[i].y;
                vect_length[i] = std::sqrt(vect[i].x*vect[i].x + vect[i].y*vect[i].y);
            }
            minarea = (right_x - left_x) * (top_y - bottom_y);
        }


        // Main loop - evaluate angles and rotate calipers

        // All of edges will be checked while rotating calipers by 90 degrees
        for(int k = 0; k < n; k++)
        {
            // The cosine of angle between calipers side and polygon edge
            float cos[4] =
            {
                ( base_a * vect[seq[0]].x + base_b * vect[seq[0]].y) / vect_length[seq[0]],
                (-base_b * vect[seq[1]].x + base_a * vect[seq[1]].y) / vect_length[seq[1]],
                (-base_a * vect[seq[2]].x - base_b * vect[seq[2]].y) / vect_length[seq[2]],
                (+base_b * vect[seq[3]].x - base_a * vect[seq[3]].y) / vect_length[seq[3]]
            };

            // rotate calipers
            {
                // The number of calipers edges that has minimal angle with edge 
                int main_element = -1;
                
                float maxcos = -1;
                // choose minimal angle 
                for(int i = 0; i < 4; ++i)
                {
                    if(cos[i] > maxcos)
                    {
                        maxcos = cos[i];
                        main_element = i;
                    }
                } 
                
                // index of the main clement point 
                int idx = seq[main_element];

                // Get next base
                switch( main_element )
                {
                    case 0:
                        base_a = vect[idx].x / vect_length[idx];
                        base_b = vect[idx].y / vect_length[idx];
                        break;
                    case 1:
                        base_a = vect[idx].y / vect_length[idx];
                        base_b = -vect[idx].x / vect_length[idx];
                        break;
                    case 2:
                        base_a = -vect[idx].x / vect_length[idx];
                        base_b = -vect[idx].y / vect_length[idx];
                        break;
                    case 3:
                        base_a = -vect[idx].y / vect_length[idx];
                        base_b =  vect[idx].x / vect_length[idx];
                        break;
                    default:
                        CV_Error(CV_StsError, "main_element should be 0, 1, 2 or 3");
                        break;
                }

                // Change base point of main edge
                if(++seq[main_element] == n)
                    seq[main_element] = 0;
            }
            
            // Find area of rectangle
            {   
                /* find vector left-right */
                float dx = points[seq[1]].x - points[seq[3]].x;
                float dy = points[seq[1]].y - points[seq[3]].y;
                float width = base_a * dx + base_b * dy;

                /* find vector bottom-top */
                dx = points[seq[0]].x - points[seq[2]].x;
                dy = points[seq[0]].y - points[seq[2]].y;
                float height = base_b * dx - base_a * dy;
                
                float area = width * height;
                if( area <= minarea )
                {
                    minarea = area;
                    
                    box.size.width = width;
                    box.size.height = height;
                    box.angle = atan2( base_b, base_a) * 180 / CV_PI ;

                    float ab = base_a * base_b;
                    float bb = base_b * base_b;
                    float aa = base_a * base_a;
                        
                    box.center.x = ( ab * (points[seq[3]].y + points[seq[1]].y - points[seq[0]].y - points[seq[2]].y)
                                 + bb * (points[seq[0]].x + points[seq[2]].x) + aa *( points[seq[1]].x + points[seq[3]].x))
                                 / (2.0 * (aa + bb));
                        
                    box.center.y = ( ab * (points[seq[3]].x + points[seq[1]].x - points[seq[0]].x - points[seq[2]].x)
                                 + bb * (points[seq[1]].y + points[seq[3]].y) + aa * (points[seq[0]].y + points[seq[2]].y) )
                                 / (2.0 * (aa + bb));
                }
            }

        }     // for ended
    }    // if n > 2 end
    else if( n == 2 )
    {
        box.center.x = (points[0].x + points[1].x)*0.5f;
        box.center.y = (points[0].y + points[1].y)*0.5f;
        float dx = points[1].x - points[0].x;
        float dy = points[1].y - points[0].y;
        box.size.width = std::sqrt(dx*dx + dy*dy);
        box.size.height = 0;
        box.angle = atan2( dy, dx );
    }
    else if( n == 1 )
        box.center = points[0];
    else
        fprintf(stderr, "rotatingCalipersMinAreaRect:no input point!\n");


    return box; 
}
