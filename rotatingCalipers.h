#ifndef ROTATINGCALIPERS_H_
#define ROTATINGCALIPERS_H_

#include <vector>
#include <cv.h>

cv::RotatedRect minAreaRect(std::vector<cv::Point> &points);
#endif
