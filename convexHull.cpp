// Implementation of Andrew's monotone chain 2D convex hull algorithm.
// Asymptotic complexity: O(n log n).
// Practical performance: 0.5-1.0 seconds for n=1000000 on a 1GHz machine.
#include <algorithm>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
// Returns a positive value, if OAB makes a counter-clockwise turn,
// negative for clockwise turn, and zero if the points are collinear.
int static cross(const cv::Point &O, const cv::Point &A, const cv::Point &B)
{
	return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}
	
bool static ptCmp(const cv::Point &p, const cv::Point &q)
{
    return p.x < q.x || (p.x == q.x && p.y < q.y);
}


// Returns a list of points on the convex hull in counter-clockwise order.
// Note: the last point in the returned list is the same as the first one.
std::vector<cv::Point> convex_hull(std::vector<cv::Point> P)
{
	int n = P.size(), k = 0;
	if (n == 1)
        return P;
    std::vector<cv::Point> H(2*n);

	// Sort points lexicographically
    std::sort(P.begin(), P.end(), ptCmp);

	// Build lower hull
	for (int i = 0; i < n; ++i)
    {
		while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0)
            k--;
		H[k++] = P[i];
	}

	// Build upper hull
	for (int i = n-2, t = k+1; i >= 0; i--)
    {
		while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0)
            k--;
		H[k++] = P[i];
	}

	H.resize(k-1);
	return H;
}



/*
int main()
{
    std::vector<Point> points;
    int nPtCnt = 50000; //生成的随机点数
    cv::Mat img(900, 1440, CV_8UC3);
    cv::RNG rng((unsigned)time(NULL));

    while(1)
    {
        img = cv::Scalar::all(0);
        points.clear();
        for (int i = 0; i < nPtCnt; ++i)
        {  
            Point pt;
            pt.x = rng.uniform(img.cols/10, img.cols*9/10);
            pt.y = rng.uniform(img.rows/10, img.rows*9/10);
            points.push_back(pt);
        }
        
        // Draw the points
        for(int i = 0; i < nPtCnt; i++ )
            cv::circle( img, cv::Point(points[i].x, points[i].y), 1, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA );

        std::vector<Point> hPts = convex_hull(points);
        
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
                for(int i = 0; i < hPts.size(); i++ )
                    printf("{%d, %d}, ", hPts[i].x, hPts[i].y);

                printf("\n");
        }
    }
    return 0;
}
*/
