#ifndef FINDCONTOURS_H_
#define FINDCONTOURS_H_


//////////////////////data structure ////////////////////////
typedef struct CvTreeNode
{
    int       flags;         /* micsellaneous flags */
    int       header_size;   /* size of sequence header */
    struct    CvTreeNode* h_prev; /* previous sequence */
    struct    CvTreeNode* h_next; /* next sequence */
    struct    CvTreeNode* v_prev; /* 2nd previous sequence */
    struct    CvTreeNode* v_next; /* 2nd next sequence */
}CvTreeNode;

typedef struct treeNodeIterator
{
    const void* node;
    int level;
    int max_level;
}treeNodeIterator;

////////////////////////////////////////////////////////////


struct Contour
{
    int       flags;            /**< Miscellaneous flags.     */
    int       header_size;      /**< Size of sequence header. */
    struct    CvSeq* h_prev;    /**< Previous sequence.       */
    struct    CvSeq* h_next;    /**< Next sequence.           */
    struct    CvSeq* v_prev;    /**< 2nd previous sequence.   */      
    struct    CvSeq* v_next;    /**< 2nd next sequence.       */


    int       total;          /**< Total number of elements.            */
    int       elem_size;      /**< Size of sequence element in bytes.   */
    schar*    block_max;      /**< Maximal bound of the last block.     */
    schar*    ptr;            /**< Current write pointer.               */
    int       delta_elems;    /**< Grow seq this many at a time.        */
    CvMemStorage* storage;    /**< Where the seq is stored.             */
    CvSeqBlock* free_blocks;  /**< Free blocks list.                    */
    CvSeqBlock* first;        /**< Pointer to the first sequence block. */

    CvRect rect;
    int color;
    int reserved[3];
};


/****************************************************************************************\
*                         Raster->Chain Tree (Suzuki algorithms)                         *
\****************************************************************************************/

typedef struct ContourInfo
{
    int flags;
    struct ContourInfo *next;    /* next contour with the same mark value */
    struct ContourInfo *parent;  /* information about parent contour */
    CvSeq *contour;              /* corresponding contour (may be 0, if rejected) */
    CvRect rect;                 /* bounding rectangle */
    CvPoint origin;              /* origin point (where the contour was traced from) */
    int is_hole;                 /* hole flag */
} ContourInfo;


/*
  Structure that is used for sequential retrieving contours from the image.
  It supports both hierarchical and plane variants of Suzuki algorithm.
*/
struct ContourScanner
{
    CvMemStorage *storage1;     /* contains fetched contours */
    CvMemStorage *storage2;     /* contains approximated contours, (!=storage1 if approx_method2 != approx_method1) */
    CvMemStorage *cinfo_storage;/* contains ContourInfo nodes */
    CvSet *cinfo_set;           /* set of ContourInfo nodes */
    CvMemStoragePos initial_pos;/* starting storage pos */
    CvMemStoragePos backup_pos; /* beginning of the latest approx. contour */
    CvMemStoragePos backup_pos2;/* ending of the latest approx. contour */
    schar *img0;                /* image origin */
    schar *img;                 /* current image row */
    int img_step;               /* image step */
    CvSize img_size;            /* ROI size */
    CvPoint offset;             /* ROI offset: coordinates, added to each contour point */
    CvPoint pt;                 /* current scanner position */
    CvPoint lnbd;               /* position of the last met contour */
    int nbd;                    /* current mark val */
    ContourInfo *l_cinfo;       /* information about latest approx. contour */
    ContourInfo cinfo_temp;     /* temporary var which is used in simple modes */
    ContourInfo frame_info;     /* information about frame */
    CvSeq frame;                /* frame itself */
    int approx_method;          /* approx method when tracing */
    int mode;                   /* contour scanning mode:
                                   0 - external only
                                   1 - all the contours w/o any hierarchy
                                   2 - connected components (i.e. two-level structure - external contours and holes),
                                   3 - full hierarchy;
                                   4 - connected components of a multi-level image */
    int seq_type;              /* type of fetched contours */
    int header_size;           /* hdr size of fetched contours */
    int elem_size;             /* elem size of fetched contours */
    
    ContourInfo *cinfo_table[128];
};

void findContours(cv::Mat& _image, std::vector< std::vector<cv::Point> >& _contours, std::vector<cv::Vec4i>& _hierarchy);

#endif
