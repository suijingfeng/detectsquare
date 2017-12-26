#include <opencv2/imgproc.hpp>

void threshold(cv::Mat &img, unsigned int iThresh, unsigned int max_value)
{
    for (int jj = 0; jj < img.rows; jj++)
    {
        unsigned char * row = img.ptr(jj);
        for (int ii = 0; ii < img.cols; ii++)
        {
            if( row[ii] <= iThresh )
                row[ii] = 0;
            else
                row[ii] = max_value;
        }
    }
}


void threshold( const CvMat* srcarr, unsigned int thresh, unsigned int maxval)
{
    cv::Mat src = cv::cvarrToMat(srcarr);
/* 
    CV_Assert( src.size == dst.size && src.channels() == dst.channels() &&
        (src.depth() == dst.depth() || dst.depth() == CV_8U));

    thresh = cv::threshold( src, dst, thresh, maxval, type );
    if( dst0.data != dst.data )
        dst.convertTo( dst0, dst0.depth() );
    return thresh;
*/
    threshold( src, thresh, maxval);
}

void biMakeBorder(cv::Mat &img, unsigned char val)
{
    int esz = img.type();
    
    int m = img.rows;
    int n = img.cols;
    int step = img.step;

    unsigned char *fRow = img.ptr(0);
    unsigned char *lRow = img.ptr(m-1);

    for (int i = 0; i < n; i++)
    {
        fRow[i] = lRow[i] = val;
    }

    unsigned char *fCol = fRow + step;
    unsigned char *lCol = fCol + step - 1; 

    for(int y = 1; y < n - 1; y++)
    {
        *fCol = *lCol = val;
        fCol += step;
        lCol += step;
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


float bimMean( cv::Mat & img )
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
