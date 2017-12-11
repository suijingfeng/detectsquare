#include <algorithm>  
#include <iostream>  
#include <vector>  
#include <math.h>  

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;  


//判断两个点(或向量)是否相等  
bool inline operator!=(const Point &pt1, const Point &pt2)
{  
    return (pt1.x != pt2.x || pt1.y != pt2.y);
}

Point inline operator-(const Point &pt1, const Point &pt2)
{  
    return Point(pt1.x - pt2.x, pt1.y - pt2.y);
}


// 比较向量中哪个与x轴向量(1, 0)的夹角更大  
bool CompareCos(const Point &pt1, const Point &pt2)
{   
    float m1 = sqrt((float)(pt1.x * pt1.x + pt1.y * pt1.y));  
    float m2 = sqrt((float)(pt2.x * pt2.x + pt2.y * pt2.y));  
    //两个向量分别与(1, 0)求内积  
    float cos1 = pt1.x / m1;
    float cos2 = pt2.x / m2;  
    //如果向量夹角相等，则返回离基点较近的一个，保证有序
    return (cos1 > cos2 || ( cos1 == cos2 && m1 < m2 ));  
}

// A utility function to swap two points
void inline swap(Point &p1, Point &p2)
{
    Point temp = p1;
    p1 = p2;
    p2 = temp;
}


//计算凸包  
void CalcConvexHull(std::vector<Point> &points, std::vector<Point> &hpoints)
{
    int n = points.size();
    int baseId;
    std::vector<Point> vec;
        
    if (n < 3)
    { 
        std::cout << "Input point less than three\n" << std::endl;
        return;  
    }

    // find the base points
    Point ptBase = points[0];
    for (int i = 1; i < n; ++i)
    {  
        //如果当前点的y值小于最小点，或y值相等，x值较小, 将当前点作为最小点
        if (points[i].y < ptBase.y || (points[i].y == ptBase.y && points[i].x > ptBase.x))
        {
            ptBase = points[i];
            baseId = i;
        }
    }

    swap(points[0], points[baseId]);

    //计算出各点与基点构成的向量  
    for (int i = 1; i < n; ++i)
        vec.push_back(points[i] - ptBase);
    
    //按各向量与横坐标之间的夹角排序  
    std::sort(vec.begin(), vec.end(), &CompareCos);

    //删除相同的向量
    vec.erase(unique(vec.begin(), vec.end()), vec.end());

    for(int i = vec.size()-1; i > 0; i--)
        vec[i] -= vec[i-1];

    //依次删除不在凸包上的向量
    for (int i = 1; i < vec.size(); ++i)
    {  
        for (int j = i - 1; j >= 0; j--)
        {  
            int cross = vec[i].x * vec[j].y - vec[i].y * vec[j].x;  
            // <=
            if (cross < 0 || (cross == 0 && vec[i].x * vec[j].x > 0 && vec[i].y * vec[j].y > 0))
                break;

            //删除前一个向量后，需更新当前向量，与前面的向量首尾相连  
            vec[i].x += vec[j].x;
            vec[i].y += vec[j].y;
            
            vec.erase(vec.begin()+j);
            i=j;
        }
    }

    //添加基点
    hpoints.push_back(ptBase);

    vec[0].x += ptBase.x;
    vec[0].y += ptBase.y;
    hpoints.push_back(vec[0]);

    //将所有首尾相连的向量换算成坐标  
    for(int i = 1; i < vec.size(); ++i)
    {  
        vec[i].x += vec[i-1].x;
        vec[i].y += vec[i-1].y;
        hpoints.push_back( vec[i] );
    }
}


int main(void)
{
    Mat img(900, 900, CV_8UC3);
    RNG rng((unsigned)time(NULL));
    
    int nPtCnt = 1000; //生成的随机点数
    
    std::vector<Point> vecSrc;
    std::vector<Point> hPts; 

    while(1)
    {
        img = Scalar::all(0);
        vecSrc.clear();
        hPts.clear();
         
        for (int i = 0; i < nPtCnt; ++i)
        {  
            Point pt;
            pt.x = rng.uniform(img.cols/10, img.cols*9/10);
            pt.y = rng.uniform(img.rows/10, img.rows*9/10);

            vecSrc.push_back(pt); 
        }

        // Draw the points
        for(int i = 0; i < vecSrc.size(); i++ )
            circle( img, vecSrc[i], 1, Scalar(0, 0, 255), FILLED, LINE_AA );


        CalcConvexHull(vecSrc, hPts);
     
        int nleft = hPts.size();
        // Draw the bounding box
        for(int i = 0; i < nleft; i++ )
            line(img, hPts[i], hPts[(i+1) % nleft], Scalar(0, 255, 0), 1, LINE_AA);

        imshow("Rectangle", img );

        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            continue;
        else if( key == 's')
        {
            for(int i = 0; i < nleft; i++ )
                printf("{%d, %d}, ", hPts[i].x, hPts[i].y);

            printf("\n");
        }
    }
    return 0;  
}  
