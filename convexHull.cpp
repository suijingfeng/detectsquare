#include <algorithm>  
#include <iostream>  
#include <vector>  
#include <math.h>  

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;  


//�ж�������(������)�Ƿ����  
bool operator==(const Point &pt1, const Point &pt2)
{  
    return (pt1.x == pt2.x && pt1.y == pt2.y);
}

// �Ƚ��������ĸ���x������(1, 0)�ļнǸ���  
bool CompareVector(const Point &pt1, const Point &pt2)
{   
    float m1 = sqrt((float)(pt1.x * pt1.x + pt1.y * pt1.y));  
    float m2 = sqrt((float)(pt2.x * pt2.x + pt2.y * pt2.y));  
    //���������ֱ���(1, 0)���ڻ�  
    float v1 = pt1.x / m1;
    float v2 = pt2.x / m2;  
    //��������н���ȣ��򷵻������Ͻ���һ������֤����
    return (v1 > v2 || ( v1 == v2 && m1 < m2 ));  
}

//����͹��  
void CalcConvexHull(std::vector<Point> &vecSrc)
{  
    //�㼯������Ӧ��3���㣬���ܹ��ɶ����  
    if (vecSrc.size() < 3)
    { 
        std::cout << "Input point less than three\n" << std::endl;
        return;  
    }

    //���һ���  
    Point ptBase = vecSrc.front(); //����1����Ԥ��Ϊ��С��
    for (std::vector<Point>::iterator i = vecSrc.begin() + 1; i != vecSrc.end(); ++i)
    {  
        //�����ǰ���yֵС����С�㣬��yֵ��ȣ�xֵ��С  
        if (i->y < ptBase.y || (i->y == ptBase.y && i->x > ptBase.x))
        {  
            ptBase = *i; //����ǰ����Ϊ��С��
        }  
    }  

    //�������������㹹�ɵ�����  
    for (std::vector<Point>::iterator i = vecSrc.begin(); i != vecSrc.end(); )
    {  
        //�ų��������ͬ�ĵ㣬����������������г��ֳ�0����  
        if (*i == ptBase)
        {  
            i = vecSrc.erase(i);  
        }
        else
        {  
            //�����ɻ��㵽Ŀ���  
            i->x -= ptBase.x, i->y -= ptBase.y;  
            ++i;  
        }
    }

    //���������������֮��ļн�����  
    sort(vecSrc.begin(), vecSrc.end(), &CompareVector);

    //ɾ����ͬ������
    vecSrc.erase(unique(vecSrc.begin(), vecSrc.end()), vecSrc.end());
    
    //����õ���β��������������
    for (std::vector<Point>::reverse_iterator ri = vecSrc.rbegin(); ri != vecSrc.rend() - 1; ++ri)
    {  
        std::vector<Point>::reverse_iterator riNext = ri + 1;  
        //���������μ��㹫ʽ  
        ri->x -= riNext->x, ri->y -= riNext->y;  
    }

    //����ɾ������͹���ϵ�����  
    for (std::vector<Point>::iterator i = vecSrc.begin() + 1; i != vecSrc.end(); ++i)
    {  
        //����ɾ����ת�����෴��������ʹ������ж���ת����  
        for (std::vector<Point>::iterator iLast = i - 1; iLast != vecSrc.begin();)
        {  
            int v1 = i->x * iLast->y, v2 = i->y * iLast->x;  
            //������С��0����û��������ת  
            //����������0�������жϷ����Ƿ�����  
            if (v1 < v2 || (v1 == v2 && i->x * iLast->x > 0 && i->y * iLast->y > 0))
            {  
                break;
            }
            //ɾ��ǰһ������������µ�ǰ��������ǰ���������β����  
            //���������μ��㹫ʽ  
            i->x += iLast->x, i->y += iLast->y;
            iLast = (i = vecSrc.erase(iLast)) - 1;
        }
    }

    //��������β���������������ۼӣ����������  
    vecSrc.front().x += ptBase.x, vecSrc.front().y += ptBase.y;
    for(std::vector<Point>::iterator i = vecSrc.begin() + 1; i != vecSrc.end(); ++i)
    {  
        i->x += (i - 1)->x, i->y += (i - 1)->y;  
    }
    //��ӻ��㣬ȫ����͹���������  
    vecSrc.push_back(ptBase);
}


int main(void)
{
    Mat img(500, 500, CV_8UC3);
    RNG rng((unsigned)time(NULL));
    
    int nPtCnt = 50; //���ɵ��������

    while(1)
    {
        img = Scalar::all(0);

        std::vector<Point> vecSrc;  
        for (int i = 0; i < nPtCnt; ++i)
        {  
            Point pt;
            pt.x = rng.uniform(img.cols/4, img.cols*3/4);
            pt.y = rng.uniform(img.rows/4, img.rows*3/4);

            vecSrc.push_back(pt); 
        }

        // Draw the points
        for(int i = 0; i < nPtCnt; i++ )
            circle( img, vecSrc[i], 3, Scalar(0, 0, 255), FILLED, LINE_AA );


        CalcConvexHull(vecSrc);
        cout << "\nConvex Hull:\n";  
        for (std::vector<Point>::iterator i = vecSrc.begin(); i != vecSrc.end(); ++i)
        {  
            cout << i->x << ", " << i->y << endl;  
        }

        int nleft = vecSrc.size();
        // Draw the bounding box
        for(int i = 0; i < nleft; i++ )
            line(img, vecSrc[i], vecSrc[(i+1) % nleft], Scalar(0, 255, 0), 1, LINE_AA);

        imshow("Rectangle", img );

        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            continue;
    }
    return 0;  
}  
