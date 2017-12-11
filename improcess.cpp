

// cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point(-1, -1)
static ContourScanner startFindContours_Impl(cv::Mat& _img, CvMemStorage* storage, int mode, int method, CvPoint offset)
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    //CvMat stub;
    //CvMat *mat = cvGetMat( _img, &stub );
    CvMat mat = _img;

    if( CV_MAT_TYPE(mat.type) == CV_32SC1 && mode == CV_RETR_CCOMP )
        mode = CV_RETR_FLOODFILL;

    uchar* img = (uchar*)(mat.data.ptr);

    ContourScanner scanner = (ContourScanner)cvAlloc( sizeof( *scanner ));
    memset( scanner, 0, sizeof(*scanner) );

    scanner->storage1 = scanner->storage2 = storage;
    scanner->img0 = (schar *) img;
    scanner->img = (schar *) (img + mat.step);
    scanner->img_step = mat.step;
    scanner->img_size.width = mat.width - 1;   /* exclude rightest column */
    scanner->img_size.height = mat.height - 1; /* exclude bottomost row */
    scanner->mode = mode;
    scanner->offset = offset;
    scanner->pt.x = scanner->pt.y = 1;
    scanner->lnbd.x = 0;
    scanner->lnbd.y = 1;
    scanner->nbd = 2;
    scanner->frame_info.contour = &(scanner->frame);
    scanner->frame_info.is_hole = 1;
    scanner->frame_info.next = 0;
    scanner->frame_info.parent = 0;
    scanner->frame_info.rect = cvRect( 0, 0, mat.width, mat.height );
    scanner->l_cinfo = 0;
    scanner->subst_flag = 0;
    scanner->frame.flags = CV_SEQ_FLAG_HOLE;
    scanner->approx_method2 = scanner->approx_method1 = method;

    scanner->seq_type2 = scanner->seq_type1 = CV_SEQ_POLYGON;
    scanner->header_size2 = scanner->header_size1 = sizeof( Contour );
    scanner->elem_size2 = scanner->elem_size1 = sizeof( CvPoint );

    scanner->seq_type1 = scanner->approx_method1 == CV_CHAIN_CODE ? CV_SEQ_CHAIN_CONTOUR : CV_SEQ_POLYGON;
    scanner->seq_type2 = scanner->approx_method2 == CV_CHAIN_CODE ? CV_SEQ_CHAIN_CONTOUR : CV_SEQ_POLYGON;

    cvSaveMemStoragePos( storage, &(scanner->initial_pos) );

    if( mode > CV_RETR_LIST )
    {
        scanner->cinfo_storage = cvCreateChildMemStorage( scanner->storage2 );
        scanner->cinfo_set = cvCreateSet( 0, sizeof( CvSet ), sizeof( _CvContourInfo ), scanner->cinfo_storage );
    }

    threshold(_img, 0, 1);
/*
    // converts all pixels to 0 or 1
    if( CV_MAT_TYPE(mat.type) != CV_32S )
    {    
        cvThreshold( mat, mat, 0, 1, CV_THRESH_BINARY );
        std::cout<<" CV_MAT_TYPE(mat->type) != CV_32S " << std::endl;
    }
*/   
    return scanner;
}


//////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////

void bimInverse(cv::Mat & img)
{
    for (int j = 0; j < img.rows; j++)
    {
        unsigned char * row = img.ptr(j);
        for (int i = 0; i < img.cols; i++)
            row[i] = ~row[i];
    }
}

///////////////////////////////////////////////////////////////////////////////////
/* 
//Copy all sequence elements into single continuous array: 
void* CvtSeqToArray(const CvSeq *seq, void *array, CvSlice slice )
{
    int elem_size, total;
    CvSeqReader reader;
    char *dst = (char*)array;

    if( !seq || !array )
        CV_Error( CV_StsNullPtr, "" );

    elem_size = seq->elem_size;
    total = cvSliceLength( slice, seq )*elem_size;

    if( total == 0 )
        return 0;

    cvStartReadSeq( seq, &reader, 0 );
    cvSetSeqReaderPos( &reader, slice.start_index, 0 );

    do
    {
        int count = (int)(reader.block_max - reader.ptr);
        if( count > total )
            count = total;

        memcpy( dst, reader.ptr, count );
        dst += count;
        reader.block = reader.block->next;
        reader.ptr = reader.block->data;
        reader.block_max = reader.ptr + reader.block->count*elem_size;
        total -= count;
    }
    while( total > 0 );

    return array;
}
*/



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


