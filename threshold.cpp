#include <cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


void threshold( cv::Mat & img, unsigned int iThresh )
{
    if( iThresh > 0 )
    {
        for (int jj = 0; jj < img.rows; jj++)
        {
            unsigned char * row = img.ptr(jj);
            for (int ii = 0; ii < img.cols; ii++)
            {
                if( row[ii] < iThresh )
                    row[ii] = 0;
                else
                    row[ii] = 255;
            }
        }
    }
}

/* binary image inverse */
void bimInverse(cv::Mat & img)
{
    for (int j = 0; j < img.rows; j++)
    {
        unsigned char * row = img.ptr(j);
        for (int i = 0; i < img.cols; i++)
        {
            row[i] = ~row[i];
        }
    }
}


float imMean( cv::Mat & img )
{
    unsigned int sum = 0;
    for (int jj = 0; jj < img.rows; jj++)
    {
        unsigned char * row = img.ptr(jj);
        for (int ii = 0; ii < img.cols; ii++)
        {
            sum += row[ii];
        }
    }
    return ( (float)sum / (img.rows * img.cols ));
}


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
    std::cout << "src.dims: " << src.dims << endl;
    std::cout << "src.isContinuous(): " << src.isContinuous() << endl;
    std::cout << "src.channels(): " << src.channels() << endl;

    std::cout << "imgGray.dims: " << imgGray.dims << endl;
    std::cout << "imgGray.isContinuous(): " << imgGray.isContinuous() << endl;
    std::cout << "imgGray.channels(): " << imgGray.channels() << endl;

    namedWindow( "Source", 1 );
    imshow( "Source", src );

    std::cout << cv::mean(src) << endl;

    Mat dst = imgGray.clone();
    mean = imMean(dst);
    std::cout << "gray image mean: " << mean << std::endl;
    threshold(dst, mean );
        
    namedWindow( "threshold", 1 );
    imshow( "threshold", dst );
    waitKey(0);
}

