// A C++ program to find convex hull of a set of points
// Refer http://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
#include <iostream>
#include <stack>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;  

typedef struct Point
{
    int x;
    int y;

    Point(int a, int b)
    {
        x = a;
        y = b;
    }
} Point;

Point p0(0, 0);

// A utility function to find next to top in a stack
Point nextToTop(stack<Point> &S)
{
    Point p = S.top();
    S.pop();
    Point res = S.top();
    S.push(p);
    return res;
}


// A utility function to return square of distance between p1 and p2
int squareDist(Point p1, Point p2)
{
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

Point inline operator-(const Point &pt1, const Point &pt2)
{  
    return Point(pt1.x - pt2.x, pt1.y - pt2.y);
}

Point inline operator+(const Point &pt1, const Point &pt2)
{  
    return Point(pt1.x + pt2.x, pt1.y + pt2.y);
}

bool inline operator!=(const Point &pt1, const Point &pt2)
{  
    return (pt1.x != pt2.x || pt1.y != pt2.y);
}

bool inline operator==(const Point &pt1, const Point &pt2)
{  
    return (pt1.x == pt2.x && pt1.y == pt2.y);
}

// 0: colinear, < 0: clock wise, > 0: counterclock wise
int orientation(const Point& p, const Point& q, const Point& r)
{
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    return val;
}
 
// A function used by library function qsort() to sort an array of points with respect to the first point
  
int compare(const void *vp1, const void *vp2)
{
    Point *p1 = (Point *) vp1;
    Point *p2 = (Point *) vp2;
 
    // Find orientation
    int o = orientation(p0, *p1, *p2);
    if (o == 0)
        return (squareDist(p0, *p2) >= squareDist(p0, *p1)) ? -1 : 1;
 
    return o;
}

/*  
// 比较向量中哪个与x轴向量(1, 0)的夹角更大  
bool CompareCos(const Point &pt1, const Point &pt2)
{   
    unsigned int m1 = pt1.x * pt1.x + pt1.y * pt1.y;  
    unsigned int m2 = pt2.x * pt2.x + pt2.y * pt2.y;  
    //两个向量分别与(1, 0)求内积  
    double cos1 = pt1.x / sqrt((double)m1);
    double cos2 = pt2.x / sqrt((double)m2);  
    //如果向量夹角相等，则返回离基点较近的一个，保证有序
    return (cos1 > cos2 || (cos1 == cos2 && m1 < m2 ));
}

bool CmpCos(const Point &pt1, const Point &pt2)
{   
    unsigned int m1 = pt1.x * pt1.x + pt1.y * pt1.y;  
    unsigned int m2 = pt2.x * pt2.x + pt2.y * pt2.y;  
    //两个向量分别与(1, 0)求内积  
    double cos1 = pt1.x / sqrt((double)m1);
    double cos2 = pt2.x / sqrt((double)m2);  
    //如果向量夹角相等，则返回离基点较近的一个，保证有序
    return (cos1 > cos2 || (cos1 == cos2 && pt1.y < pt2.y) );
}

*/

// Prints convex hull of a set of n points.
void convexHull(std::vector<Point> &points, std::vector<Point> &hull)
{
    // Find the bottommost point
    int ymin = points[0].y;
    int minId = 0;
    
    int n = points.size();
    for (int i = 1; i < n; i++)
    {
        int y = points[i].y;
 
        // Pick the bottom-most or chose the left most point in case of tie
        if ((y < ymin) || ( y == ymin && points[i].x < points[minId].x))
        {
            ymin = y;
            minId = i;
        }
    }
    
    //Place the bottom-most point at first position
    Point p = points[0]; 
    points[0] = points[minId]; 
    points[minId] = p;
 
    p0 = points[0];

    qsort(&points[1], n - 1, sizeof(Point), compare);
 
   // If two or more points make same angle with p0,
   // Remove all but the one that is farthest from p0
   // Remember that, in above sorting, our criteria was
   // to keep the farthest point at the end when more than
   // one points have same angle.
   int m = 1; // Initialize size of modified array
   for (int i=1; i<n; i++)
   {
       // Keep removing i while angle of i and i+1 is same
       // with respect to p0
       while (i < n-1 && orientation(p0, points[i], points[i+1]) == 0)
          i++;
 
       // Update size of modified array
       points[m++] = points[i];
   }
   if (m < 3) 
        return;
    // Create an empty stack and push first three points to it.
    stack<Point> S;
    S.push(points[0]);
    S.push(points[1]);
    S.push(points[2]);
 
    // Process remaining n-3 points
    for(int i = 3; i < m; i++)
    {
        // Keep removing top while the angle formed by points next-to-top,
        // top, and points[i] makes a non-left turn
        try{
            
            while (orientation(nextToTop(S), S.top(), points[i]) >= 0)
                S.pop();

            S.push(points[i]);
        }
        catch (const std::exception& e)
        { // reference to the base of a polymorphic object
            std::cout << e.what(); // information from length_error printed
        }
    }
 
    // Now stack has the output points, print contents of stack
    while (!S.empty())
    {
        hull.push_back(S.top());
        S.pop();
    }
}


// Driver program to test above functions
int main()
{
    std::vector<Point> points;
    cv::Mat img(900, 900, CV_8UC3);
    cv::RNG rng((unsigned)time(NULL));
    std::vector<Point> hPts; 

    int nPtCnt = 5000; //生成的随机点数

    while(1)
    {
        points.clear();
        hPts.clear();
        img = cv::Scalar::all(0);

        for (int i = 0; i < nPtCnt; ++i)
        {  
            Point pt( rng.uniform(img.cols/10, img.cols*9/10), rng.uniform(img.rows/10, img.rows*9/10));
            points.push_back(pt);
        }
        
        // Draw the points
        for(int i = 0; i < nPtCnt; i++ )
            cv::circle( img, cv::Point(points[i].x, points[i].y), 1, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA );

        convexHull(points, hPts);
        
        int nleft = hPts.size();
            // Draw the bounding box
        for(int i = 0; i < nleft; i++ )
            cv::line(img, cv::Point(hPts[i].x, hPts[i].y) , cv::Point(hPts[(i+1) % nleft].x, hPts[(i+1) % nleft].y) , cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        
        cv::imshow("Rectangle", img );

        char key = (char)cv::waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            continue;

        else if( key == 's')
        {
                for(int i = 0; i < points.size(); i++ )
                    printf("{%d, %d}, ", points[i].x, points[i].y);

                printf("\n");
        }
    }
    
    return 0;

}
