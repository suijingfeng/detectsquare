#ifndef CONVEXHULL_H_
#define CONVEXHULL_H_

#include <algorithm>
#include <vector>
#include <opencv2/imgproc.hpp>

std::vector<cv::Point> convex_hull(std::vector<cv::Point> P);

#endif
