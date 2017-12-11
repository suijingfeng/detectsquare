#ifndef THRESHOLD_H_
#define THRESHOLD_H_

void threshold(cv::Mat & img, unsigned int iThresh, unsigned int max_value);
void threshold(const CvMat* srcarr, unsigned int thresh, unsigned int maxval);
float bimMean( cv::Mat & img );
void biMakeBorder(cv::Mat &img, unsigned char val);

#endif
