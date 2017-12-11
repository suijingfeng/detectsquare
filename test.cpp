//for testing verious functions
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "threshold.h"

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    float mean = 128.0f;
    Mat src, imgGray;
    // the first command-line parameter must be a filename of the binary
    // (black-n-white) image
    if( argc != 2 )
    {
        printf("usage: ./%s imageName.png", argv[0]);
        return -1;
    }

    src = cv::imread(argv[1], 1);
    if (src.channels() != 1)
    {
        cvtColor(src, imgGray, cv::COLOR_BGR2GRAY);
    }
    cv::Size imageSize = imgGray.size();
/*  
    std::cout << "src.dims: " << src.dims << endl;
    std::cout << "src.isContinuous(): " << src.isContinuous() << endl;
    std::cout << "src.channels(): " << src.channels() << endl;

    std::cout << "imgGray.dims: " << imgGray.dims << endl;
    std::cout << "imgGray.isContinuous(): " << imgGray.isContinuous() << endl;
    std::cout << "imgGray.channels(): " << imgGray.channels() << endl;
*/
    //namedWindow( "Source", 1 );
    //imshow( "Source", src );

    std::cout << cv::mean(src) << endl;

    Mat dst1 = imgGray.clone();
    Mat dst2 = imgGray.clone();

    //mean = bimMean(dst1);
    //std::cout << "gray image mean: " << mean << std::endl;
    //threshold(dst1, mean, 255);
    
    biMakeBorder(dst1, 0);
    biMakeBorder(dst2, 255);

    namedWindow("dst1", 1);
    imshow("dst1", dst1);

    namedWindow("dst2", 2);
    imshow("dst2", dst2);


    waitKey(0);
}
