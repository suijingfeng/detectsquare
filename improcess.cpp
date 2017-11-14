
void bimInverse(cv::Mat & img)
{
    for (int j = 0; j < img.rows; j++)
    {
        unsigned char * row = img.ptr(j);
        for (int i = 0; i < img.cols; i++)
            row[i] = ~row[i];
    }
}

/*
int checkChessboardBinary(const cv::Mat & img, const cv::Size & size)
{
    CV_Assert(img.channels() == 1 && img.depth() == CV_8U);
    #ifdef SAVE_TEMP_RES
        imwrite("img_before_morphology.png", img);
    #endif

    cv::Mat white = img.clone();
    cv::Mat black = img.clone();

    for(int i = 0; i <= 3; i++)
    {
        if( 0 != i ) // first iteration keeps original images
        {
            erode(white, white, cv::Mat(), cv::Point(-1, -1), 1);
            dilate(black, black, cv::Mat(), cv::Point(-1, -1), 1);

            #ifdef SAVE_TEMP_RES
                std::ostringstream str;
                str << i;
                imwrite("erode_white" + str.str() + ".png", white);
                imwrite("dilate_black" + str.str() + ".png", black);
            #endif
        }
        
        vector<pair<float, int> > quads;
    
        std::vector< std::vector<cv::Point> > contours;
        std::vector< cv::Vec4i > hierarchy;

        findContours(white, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        GetQuadrangleHypotheses(contours, hierarchy, quads, 1);
        
        contours.clear();
        hierarchy.clear();
        
        cv::Mat thresh = black.clone();
        
        bimInverse(thresh);

        findContours(thresh, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        GetQuadrangleHypotheses(contours, hierarchy, quads, 0);
    
        if ( checkQuads(quads, size) )
            return 1;
    }
    return 0;
}
*/


static bool checkQuads(vector<pair<float, int> > & quads, const cv::Size & size)
{
    const unsigned int min_quads_count = size.width*size.height/2;
    const float size_rel_para = 1.4f;

    // both black and write half of the total number of quads
    //std::cout << "min_quads_count:" << min_quads_count << std::endl;
    std::sort(quads.begin(), quads.end(), less_pred);

    // now check if there are many hypotheses with similar sizes
    // do this by floodfill-style algorithm
    for(unsigned int i = 0; i < quads.size(); i++)
    {
        unsigned int j = i + 1;
        for(; j < quads.size(); j++)
        {
            if(quads[j].first/quads[i].first > size_rel_para)
            {
                break;
            }
        }

        if(j - i + 1 > min_quads_count )
        {
            // check the number of black and white squares
            std::vector<int> counts(2, 0);
            //countClasses(quads, i, j, counts);

            for(unsigned int n = i; n != j; n++)
            {
                counts[quads[n].second]++;
            }

            //const int white_count = cvRound( floor(size.width/2.0) * floor(size.height/2.0) );
            if(counts[0] < min_quads_count / 2 || counts[1] < min_quads_count / 2)
                continue;

            return true;
        }
    }
    return false;
}



int checkVector2(cv::Mat &mat, int _elemChannels, int _depth, bool _requireContinuous)
{
    if( (mat.depth() == _depth || _depth <= 0) && (mat.isContinuous() || !_requireContinuous) )
    {
        if(mat.dims == 2)
        {
            if( (( mat.rows == 1 || mat.cols == 1) && mat.channels() == _elemChannels) || (mat.cols == _elemChannels && mat.channels() == 1))
            {
                return (int)(mat.total() * mat.channels()/_elemChannels);
            }
        }
        else if( mat.dims == 3 )
        {
            if( mat.channels() == 1 && mat.size.p[2] == _elemChannels && ( mat.size.p[0] == 1 || mat.size.p[1] == 1)
                && ( mat.isContinuous() || mat.step.p[1] == mat.step.p[2] * mat.size.p[2] ) )
            {
                return (int)(mat.total() * mat.channels()/_elemChannels);
            }
        }
    }
    return -1;
}
  
void drawBox(cv::RotatedRect box)
{
    cv::Mat image(553, 478, CV_8UC3, cv::Scalar(0));
    cv::Point2f vertices[4];
    box.points(vertices);
    
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));

    imshow("rectangles", image);
    cv::waitKey(0);
}



void GetQuadrangleHypotheses(const std::vector<std::vector< cv::Point > > & contours, 
        const std::vector< cv::Vec4i > & hierarchy, std::vector<std::pair<float, int> >& quads, int class_id)
{
    const float min_aspect_ratio = 0.5f, max_aspect_ratio = 2.0f;
    const float min_quad_size = 10.0f, max_quad_pixel = 250;
    
    std::vector< std::vector< cv::Point > >::const_iterator pt;
    std::cout << "contours.size(): " << contours.size() << std::endl;

    for(pt = contours.begin(); pt != contours.end(); ++pt)
    {
        const std::vector< std::vector< cv::Point > >::const_iterator::difference_type idx = pt - contours.begin();
        
        // if the parent contour not equal to -1, then it is a hole
        //if(hierarchy.at(idx)[3] != -1)
        if(hierarchy[idx][3] != -1 || hierarchy[idx][2] != -1)
            continue; // skip holes
        
        // too small, min_box_size * 4
        if(pt->size() < 40)
            continue;
        //const std::vector< cv::Point > & c = *i;
        std::cout << "pt->size(): " << pt->size() << std::endl;
        
        cv::RotatedRect box = minRectArea(*pt);

        float box_size = MAX(box.size.width, box.size.height);
        
        if(box_size < min_quad_size || box_size > max_quad_pixel)
            continue;

        float aspect_ratio = box.size.width / MAX(box.size.height, 1);
        if(aspect_ratio < min_aspect_ratio || aspect_ratio > max_aspect_ratio)
            continue;
        
        drawBox(g_img, box);
        quads.push_back( std::pair<float, int>(box_size, class_id) );
    }
    std::cout << "quads.size(): " << quads.size() << std::endl;
}

inline bool less_pred(const std::pair<float, int>& p1, const std::pair<float, int>& p2)
{
    return p1.first < p2.first;
}




cv::RotatedRect cv::minAreaRect( InputArray _points )
{
    Mat hull;
    Point2f out[3];
    RotatedRect box;

    convexHull(_points, hull, true, true);

    if( hull.depth() != CV_32F )
    {
        Mat temp;
        hull.convertTo(temp, CV_32F);
        hull = temp;
    }

    int n = hull.checkVector(2);
    const Point2f* hpoints = hull.ptr<Point2f>();

    if( n > 2 )
    {
        rotatingCalipers( hpoints, n, CALIPERS_MINAREARECT, (float*)out );
        box.center.x = out[0].x + (out[1].x + out[2].x)*0.5f;
        box.center.y = out[0].y + (out[1].y + out[2].y)*0.5f;
        box.size.width = (float)std::sqrt((double)out[1].x*out[1].x + (double)out[1].y*out[1].y);
        box.size.height = (float)std::sqrt((double)out[2].x*out[2].x + (double)out[2].y*out[2].y);
        box.angle = (float)atan2( (double)out[1].y, (double)out[1].x );
    }
    else if( n == 2 )
    {
        box.center.x = (hpoints[0].x + hpoints[1].x)*0.5f;
        box.center.y = (hpoints[0].y + hpoints[1].y)*0.5f;
        double dx = hpoints[1].x - hpoints[0].x;
        double dy = hpoints[1].y - hpoints[0].y;
        box.size.width = (float)std::sqrt(dx*dx + dy*dy);
        box.size.height = 0;
        box.angle = (float)atan2( dy, dx );
    }
    else
    {
        if( n == 1 )
            box.center = hpoints[0];
    }

    box.angle = (float)(box.angle*180/CV_PI);
    return box;
}

//THRESHOLD


