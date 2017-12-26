#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#include "findContours.h"
#include "threshold.h"

// using namespace std;
// #define CV_GET_WRITTEN_ELEM( writer ) ((writer).ptr - (writer).seq->elem_size)

#define WRITE_SEQ_ELEM( elem, writer )             \
{                                                     \
    assert( (writer).seq->elem_size == sizeof(elem)); \
    if( (writer).ptr >= (writer).block_max )          \
    {                                                 \
        cvCreateSeqBlock( &writer);                   \
    }                                                 \
    assert( (writer).ptr <= (writer).block_max - sizeof(elem));\
    memcpy((writer).ptr, &(elem), sizeof(elem));      \
    (writer).ptr += sizeof(elem);                     \
}

// Initializes 8-element array for fast access to 3x3 neighborhood of a pixel
#define  INIT_3X3_DELTAS( deltas, step )     \
    ((deltas)[0] =  1, (deltas)[1] = -(step) + 1, (deltas)[2] = -(step), (deltas)[3] = -(step) - 1,  \
     (deltas)[4] = -1, (deltas)[5] =  (step) - 1, (deltas)[6] = (step), (deltas)[7] = (step) + 1 )

static const CvPoint icvCodeDeltas[8] =
    { CvPoint(1, 0), CvPoint(1, -1), CvPoint(0, -1), CvPoint(-1, -1), CvPoint(-1, 0), CvPoint(-1, 1), CvPoint(0, 1), CvPoint(1, 1) };


#if CV_SSE2
static inline unsigned int trailingZeros(unsigned int value)
{
    CV_DbgAssert(value != 0); // undefined for zero input (https://en.wikipedia.org/wiki/Find_first_set)
    #if defined(__GNUC__) || defined(__GNUG__)
        return __builtin_ctz(value);
    #elif defined(__ICC) || defined(__INTEL_COMPILER)
        return _bit_scan_forward(value);
    #elif defined(__clang__)
        return llvm.cttz.i32(value, true);
    #else
        #error NOT IMPLEMENTED
    #endif
}
#endif

void* nextTreeNode(treeNodeIterator* treeIterator)
{
    CvTreeNode* prevNode = 0;
    CvTreeNode* node;

    if( !treeIterator )
        CV_Error( CV_StsNullPtr, "NULL iterator pointer" );

    prevNode = node = (CvTreeNode*)treeIterator->node;

    if( node )
    {
        if( node->v_next && (treeIterator->level < treeIterator->max_level - 1) )
        {
            node = node->v_next;
            treeIterator->level++;
        }
        else
        {
            while( node->h_next == 0 )
            {
                node = node->v_prev;
                if( --treeIterator->level < 0 )
                {
                    node = 0;
                    break;
                }
            }
            node = node && treeIterator->max_level != 0 ? node->h_next : 0;
        }
    }

    treeIterator->node = node;
    return prevNode;
}



CvSeq* treeToNodeSeq(const void* first, int header_size, CvMemStorage* storage )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );
    
    CvSeq* allseq = cvCreateSeq(0, header_size, sizeof(first), storage);
    treeNodeIterator iterator;
    
    if( first )
    {
        iterator.node = first;
        iterator.level = 0;
        iterator.max_level = INT_MAX;

        for(;;)
        {
            void* node = nextTreeNode( &iterator );
            if( !node )
                break;
            cvSeqPush( allseq, &node ); //
        }
    }
    return allseq;
}

/* Calls icvFlushSeqWriter and finishes writing process: */
#define  STRUCT_ALIGN    ((int)sizeof(double))
CvSeq *cvEndWriteSeq( CvSeqWriter * writer )
{
    if( !writer )
        CV_Error( CV_StsNullPtr, "" );

    cvFlushSeqWriter( writer );
    CvSeq* seq = writer->seq;

    /* Truncate the last block: */
    if( writer->block && writer->seq->storage )
    {
        CvMemStorage *storage = seq->storage;
        schar *storage_block_max = (schar *) storage->top + storage->block_size;

        assert( writer->block->count > 0 );

        if( (unsigned)((storage_block_max - storage->free_space) - seq->block_max) < STRUCT_ALIGN )
        {
            storage->free_space = cvAlignLeft((int)(storage_block_max - seq->ptr), STRUCT_ALIGN);
            seq->block_max = seq->ptr;
        }
    }

    writer->ptr = 0;
    return seq;
}

/*  
static void changeSeqBlock(void* _reader, int direction)
{
    CvSeqReader* reader = (CvSeqReader*)_reader;

    if( !reader )
        CV_Error( CV_StsNullPtr, "" );

    if( direction > 0 )
    {
        reader->block = reader->block->next;
        reader->ptr = reader->block->data;
    }
    else
    {
        reader->block = reader->block->prev;
        // Change the current reading block to the previous
        reader->ptr = reader->block->data + (reader->block->count - 1)*(reader->seq->elem_size);
    }
    reader->block_min = reader->block->data;
    reader->block_max = reader->block_min + reader->block->count * reader->seq->elem_size;
}
*/

/* Copy all sequence elements into single std::vector */
void CvtSeqToVector(const CvSeq *seq, std::vector<cv::Point> &p)
{
    CvSeqReader reader;

    int total = seq->total; // *elem_size();
    if( total == 0 )
        return;
    p.resize(total);
    cvStartReadSeq(seq, &reader, 0);
    cv::Point elem;
    for (int i = 0; i<seq->total; i++)
    {
        assert( reader.seq->elem_size == sizeof(elem));
        memcpy( &(elem), (reader).ptr, sizeof(elem));
        p[i] = elem;

        // Move reader position forward
        reader.ptr += sizeof(elem);
        if( reader.ptr >= reader.block_max )
        {
            //changeSeqBlock( &reader, 1 );
            reader.block = (reader.block)->next;
            reader.ptr = (reader.block)->data;
            reader.block_min = (reader.block)->data;
            reader.block_max = reader.block_min + (reader.block)->count * (reader.seq)->elem_size;
        }
    }
}


/* 
   trace contour until certain point is met. returns 1 if met, 0 else.
*/
static int TraceContour( schar *ptr, int step, schar *stop_ptr, int is_hole )
{
    int deltas[16];
    schar *i0 = ptr, *i1, *i3, *i4;
    int s, s_end;

    /* initialize local state */
    INIT_3X3_DELTAS(deltas, step);
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    assert( (*i0 & -2) != 0 );

    s_end = s = (is_hole ? 0 : 4);

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    i3 = i0;

    /* check single pixel domain */
    if( s != s_end )
    {
        /* follow border */
        for( ;; )
        {
            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( *i4 != 0 )
                    break;
            }

            if( i3 == stop_ptr || (i4 == i0 && i3 == i1) )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        } /* end of border following loop */
    }
    return i3 == stop_ptr;
}



/*
    marks domain border with +/-<constant> and stores the contour into CvSeq.
        method:
            <0  - chain
            ==0 - direct
            >0  - simple approximation
*/
static void FetchContour(schar *ptr, int step, CvPoint pt, CvSeq* contour, int _method)
{
    const schar nbd = 2;
    int deltas[16];
    CvSeqWriter writer;
    schar *i0 = ptr, *i1, *i3, *i4 = 0;
    int prev_s = -1, s, s_end;
    int method = _method - 1;

    assert( (unsigned) _method <= CV_CHAIN_APPROX_SIMPLE );

    /* initialize local state */
    INIT_3X3_DELTAS( deltas, step);
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    /* initialize writer */
    cvStartAppendToSeq( contour, &writer );

    if( method < 0 )
        ((CvChain *) contour)->origin = pt;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = (schar) (nbd | -128);
        if( method >= 0 )
        {
            WRITE_SEQ_ELEM(pt, writer);
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            s_end = s;

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( *i4 != 0 )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = (schar) (nbd | -128);
            }
            else if( *i3 == 1 )
            {
                *i3 = nbd;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;
                WRITE_SEQ_ELEM( _s, writer );
            }
            else
            {
                if( s != prev_s || method == 0 )
                {
                    WRITE_SEQ_ELEM( pt, writer );
                    prev_s = s;
                }

                pt.x += icvCodeDeltas[s].x;
                pt.y += icvCodeDeltas[s].y;
            }

            if( i4 == i0 && i3 == i1 )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    cvEndWriteSeq( &writer );

    if( _method != CV_CHAIN_CODE )
        cvBoundingRect( contour, 1 );

    assert( (writer.seq->total == 0 && writer.seq->first == 0) ||
            writer.seq->total > writer.seq->first->count ||
            (writer.seq->first->prev == writer.seq->first &&
             writer.seq->first->next == writer.seq->first) );
}


static void icvFetchContourEx( schar* ptr, int step, CvPoint pt, CvSeq* contour, int _method, int nbd, CvRect* _rect)
{
    int         deltas[16];
    CvSeqWriter writer;
    schar       *i0 = ptr, *i1, *i3, *i4;
    CvRect      rect;
    int         prev_s = -1, s, s_end;
    int         method = _method - 1;

    assert( (unsigned) _method <= CV_CHAIN_APPROX_SIMPLE );
    assert( 1 < nbd && nbd < 128 );

    /* initialize local state */
    INIT_3X3_DELTAS(deltas, step);
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    /* initialize writer */
    cvStartAppendToSeq( contour, &writer );

    if( method < 0 )
        ((CvChain *)contour)->origin = pt;

    rect.x = rect.width = pt.x;
    rect.y = rect.height = pt.y;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = (schar) (nbd | 0x80);
        if( method >= 0 )
        {
            WRITE_SEQ_ELEM( pt, writer );
        }
    }
    else
    {
        i3 = i0;

        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            s_end = s;

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( *i4 != 0 )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = (schar) (nbd | 0x80);
            }
            else if( *i3 == 1 )
            {
                *i3 = (schar) nbd;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;
                WRITE_SEQ_ELEM( _s, writer );
            }
            else if( s != prev_s || method == 0 )
            {
                WRITE_SEQ_ELEM( pt, writer );
            }

            if( s != prev_s )
            {
                /* update bounds */
                if( pt.x < rect.x )
                    rect.x = pt.x;
                else if( pt.x > rect.width )
                    rect.width = pt.x;

                if( pt.y < rect.y )
                    rect.y = pt.y;
                else if( pt.y > rect.height )
                    rect.height = pt.y;
            }

            prev_s = s;
            pt.x += icvCodeDeltas[s].x;
            pt.y += icvCodeDeltas[s].y;

            if( i4 == i0 && i3 == i1 )  break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;

    cvEndWriteSeq( &writer );

    if( _method != CV_CHAIN_CODE )
        ((Contour*)contour)->rect = rect;

    assert( (writer.seq->total == 0 && writer.seq->first == 0) ||
            writer.seq->total > writer.seq->first->count ||
            (writer.seq->first->prev == writer.seq->first &&
             writer.seq->first->next == writer.seq->first) );

    if( _rect )
        *_rect = rect;
}



static void endProcessContour(struct ContourScanner* scanner)
{
    ContourInfo *l_cinfo = scanner->l_cinfo;
    if( l_cinfo )
    {
        if( l_cinfo->contour )
            cvInsertNodeIntoTree( l_cinfo->contour, l_cinfo->parent->contour, &(scanner->frame) );
        
        scanner->l_cinfo = 0;
    }
}


CvSeq* FindNextContour(struct ContourScanner* scanner)
{
    if( !scanner )
        CV_Error( CV_StsNullPtr, "" );

#if CV_SSE2
    bool haveSIMD = cv::checkHardwareSupport(CPU_SSE2);
#endif

    CV_Assert(scanner->img_step >= 0);

    endProcessContour( scanner );

    /* initialize local state */
    schar* img0 = scanner->img0;
    schar* img = scanner->img;
    int step = scanner->img_step;
    int x = scanner->pt.x;
    int y = scanner->pt.y;
    int width = scanner->img_size.width;
    int height = scanner->img_size.height;
    int mode = scanner->mode;
    CvPoint lnbd = scanner->lnbd;
    int nbd = scanner->nbd;
    int prev = img[x - 1];

    for( ; y < height; y++, img += step )
    {
        int p = 0;

        for( ; x < width; x++ )
        {
        #if CV_SSE2
            if ((p = img[x]) != prev) 
                goto _next_contour;
            else if (haveSIMD)
            {
                __m128i v_prev = _mm_set1_epi8((char)prev);
                int v_size = width - 32;

                for (; x <= v_size; x += 32) {
                    __m128i v_p1 = _mm_loadu_si128((const __m128i*)(img + x));
                    __m128i v_p2 = _mm_loadu_si128((const __m128i*)(img + x + 16));

                    __m128i v_cmp1 = _mm_cmpeq_epi8(v_p1, v_prev);
                    __m128i v_cmp2 = _mm_cmpeq_epi8(v_p2, v_prev);

                    unsigned int mask1 = _mm_movemask_epi8(v_cmp1);
                    unsigned int mask2 = _mm_movemask_epi8(v_cmp2);

                    mask1 ^= 0x0000ffff;
                    mask2 ^= 0x0000ffff;

                    if (mask1) {
                        p = img[(x += trailingZeros(mask1))];
                        goto _next_contour;
                    }

                    if (mask2) {
                        p = img[(x += trailingZeros(mask2 << 16))];
                        goto _next_contour;
                    }
                }

                if(x <= width - 16) {
                    __m128i v_p = _mm_loadu_si128((__m128i*)(img + x));

                    unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v_p, v_prev)) ^ 0x0000ffff;

                    if (mask) {
                        p = img[(x += trailingZeros(mask))];
                        goto _next_contour;
                    }
                    x += 16;
                }
            }
            #endif
            
            for( ; x < width && (p = img[x]) == prev; x++)
                ;

            if( x >= width )
                break;
#if CV_SSE2
        _next_contour:
#endif
            {
                ContourInfo *par_info = 0;
                ContourInfo *l_cinfo = 0;
                CvSeq *seq = 0;
                int is_hole = 0;
                CvPoint origin;

                /* if not external contour */
                if( !(prev == 0 && p == 1) )
                {
                    /* check hole */
                    if( p != 0 || prev < 1 )
                        goto resume_scan;

                    if( prev & -2 )
                        lnbd.x = x - 1;
                    
                    is_hole = 1;
                }

                if( mode == 0 && (is_hole || img0[lnbd.y * static_cast<size_t>(step) + lnbd.x] > 0) )
                    goto resume_scan;

                origin.y = y;
                origin.x = x - is_hole;

                /* find contour parent */
                if( mode <= 1 || (!is_hole && (mode == CV_RETR_CCOMP )) || lnbd.x <= 0 )
                {
                    par_info = &(scanner->frame_info);
                }
                else
                {
                    int lval = (int)img0[lnbd.y * static_cast<size_t>(step) + lnbd.x] & 0x7f;
                    ContourInfo *cur = scanner->cinfo_table[lval];

                    /* find the first bounding contour */
                    while( cur )
                    {
                        if( (unsigned) (lnbd.x - cur->rect.x) < (unsigned) cur->rect.width &&
                            (unsigned) (lnbd.y - cur->rect.y) < (unsigned) cur->rect.height )
                        {
                            if( par_info )
                            {
                                if( TraceContour( img0 + par_info->origin.y * static_cast<size_t>(step) + par_info->origin.x, 
                                                    step, img + lnbd.x, par_info->is_hole ) > 0 )
                                    break;
                            }
                            par_info = cur;
                        }
                        cur = cur->next;
                    }

                    assert( par_info != 0 );

                    /* if current contour is a hole and previous contour is a hole or
                       current contour is external and previous contour is external then
                       the parent of the contour is the parent of the previous contour else
                       the parent is the previous contour itself. */
                    if( par_info->is_hole == is_hole )
                    {
                        par_info = par_info->parent;
                        /* every contour must have a parent
                           (at least, the frame of the image) */
                        if( !par_info )
                            par_info = &(scanner->frame_info);
                    }

                    /* hole flag of the parent must differ from the flag of the contour */
                    assert( par_info->is_hole != is_hole );
                    if( par_info->contour == 0 )        /* removed contour */
                        goto resume_scan;
                }

                lnbd.x = x - is_hole;

                cvSaveMemStoragePos( scanner->storage2, &(scanner->backup_pos) );

                seq = cvCreateSeq( scanner->seq_type, scanner->header_size, scanner->elem_size, scanner->storage1 );
                seq->flags |= is_hole ? CV_SEQ_FLAG_HOLE : 0;

                /* initialize header */
                if( mode <= 1 )
                {
                    l_cinfo = &(scanner->cinfo_temp);
                    FetchContour( img + x - is_hole, step, cvPoint( origin.x + scanner->offset.x, origin.y + scanner->offset.y), seq, scanner->approx_method);
                }
                else
                {
                    union {
                        ContourInfo* ci;
                        CvSetElem* se;
                    } v;

                    v.ci = l_cinfo;
                    cvSetAdd( scanner->cinfo_set, 0, &v.se );
                    l_cinfo = v.ci;
                    int lval = nbd;
                    
                    // change nbd
                    nbd = (nbd + 1) & 127;
                    nbd += nbd == 0 ? 3 : 0;
                    icvFetchContourEx( img + x - is_hole, step, cvPoint( origin.x + scanner->offset.x, origin.y + scanner->offset.y),
                                        seq, scanner->approx_method, lval, &(l_cinfo->rect) );
                    
                    l_cinfo->rect.x -= scanner->offset.x;
                    l_cinfo->rect.y -= scanner->offset.y;

                    l_cinfo->next = scanner->cinfo_table[lval];
                    scanner->cinfo_table[lval] = l_cinfo;
                }

                l_cinfo->is_hole = is_hole;
                l_cinfo->contour = seq;
                l_cinfo->origin = origin;
                l_cinfo->parent = par_info;
                l_cinfo->contour->v_prev = l_cinfo->parent->contour;

                if( par_info->contour == 0 )
                {
                    l_cinfo->contour = 0;
                    if( scanner->storage1 == scanner->storage2 )
                        cvRestoreMemStoragePos( scanner->storage1, &(scanner->backup_pos) );
                    else
                        cvClearMemStorage( scanner->storage1 );
                    
                    p = img[x];
                    goto resume_scan;
                }

                cvSaveMemStoragePos( scanner->storage2, &(scanner->backup_pos2) );
                scanner->l_cinfo = l_cinfo;
                scanner->pt.x = x + 1;
                scanner->pt.y = y;
                scanner->lnbd = lnbd;
                scanner->img = (schar *) img;
                scanner->nbd = nbd;
                return l_cinfo->contour;

            resume_scan:

                prev = p;
                /* update lnbd */
                if( prev & -2 )
                    lnbd.x = x;
            }                   /* end of prev != p */
        }                       /* end of loop on x */

        lnbd.x = 0;
        lnbd.y = y + 1;
        x = 1;
        prev = 0;
    }                           /* end of loop on y */

    return 0;
}


/* The function add to tree the last retrieved/substituted contour,
   releases temp_storage, restores state of dst_storage (if needed),
   and returns pointer to root of the contour tree */

static CvSeq* endFindContours(struct ContourScanner** _scanner)
{
    struct ContourScanner* scanner;
    CvSeq *first = 0;

    if( !_scanner )
        CV_Error( CV_StsNullPtr, "" );
    scanner = *_scanner;

    if( scanner )
    {
        endProcessContour( scanner );

        if( scanner->storage1 != scanner->storage2 )
            cvReleaseMemStorage( &(scanner->storage1) );

        if( scanner->cinfo_storage )
            cvReleaseMemStorage( &(scanner->cinfo_storage) );

        first = scanner->frame.v_next;
        cvFree( _scanner );
    }

    return first;
}


/*
 * FindContours supports only CV_8UC1 images when mode != CV_RETR_FLOODFILL,
 * otherwise supports CV_32SC1 images only
 * cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point(-1, -1)
 * Initializes scanner structure, clear borders and convert all pixels to 0-1.
 */
static struct ContourScanner* startFindContours(cv::Mat& img, CvMemStorage* storage, int mode, int method, CvPoint offset)
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    struct ContourScanner* scanner = (struct ContourScanner*)cvAlloc( sizeof( *scanner ));
    memset( scanner, 0, sizeof(*scanner) );

    scanner->storage1 = scanner->storage2 = storage;
    scanner->img0 = (schar *) img.ptr(0);
    scanner->img = (schar *) img.ptr(1);
    scanner->img_step = img.step;
    scanner->img_size.width = img.cols - 1;   /* exclude rightest column */
    scanner->img_size.height = img.rows - 1; /* exclude bottomost row */
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
    scanner->frame_info.rect = cvRect( 0, 0, img.cols, img.rows );
    scanner->l_cinfo = 0;
    scanner->frame.flags = CV_SEQ_FLAG_HOLE;

    scanner->approx_method = method;

    scanner->seq_type = CV_SEQ_POLYGON;
    scanner->header_size = sizeof( struct Contour );
    scanner->elem_size = sizeof( CvPoint );

    cvSaveMemStoragePos( storage, &(scanner->initial_pos) );

/* 
    if( method > CV_CHAIN_APPROX_SIMPLE ) // method = CHAIN_APPROX_SIMPLE CV_CHAIN_APPROX_SIMPLE = 2
    {
        scanner->storage1 = cvCreateChildMemStorage( scanner->storage2 );
    }
*/
    
    if( mode > CV_RETR_LIST ) // CV_RETR_LIST =1, CV_RETR_CCOMP =2 
    {
        scanner->cinfo_storage = cvCreateChildMemStorage( scanner->storage2 );
        scanner->cinfo_set = cvCreateSet( 0, sizeof( CvSet ), sizeof( ContourInfo ), scanner->cinfo_storage );
    }

    // converts all pixels to 0 or 1
    threshold(img, 0, 1);
    return scanner;
}


static int findContoursImpl(cv::Mat& image, CvMemStorage* storage, CvSeq** firstContour, int mode, int method)
{
    struct ContourScanner* scanner = 0;
    CvSeq *contour = 0;
    int count = -1;

    if( !firstContour )
        CV_Error( CV_StsNullPtr, "NULL double CvSeq pointer" );

    *firstContour = 0;
    
    try
    {
        scanner = startFindContours(image, storage, mode, method, cv::Point(0, 0));
        do
        {
            count++;
            contour = FindNextContour( scanner );
        }
        while( contour != 0 );
    }
    catch(...)
    {
        if( scanner )
            endFindContours(&scanner);
        throw;
    }

    *firstContour = endFindContours( &scanner );
    
    return count;
}


void findContours(cv::Mat& image, std::vector< std::vector<cv::Point> >& _contours, std::vector<cv::Vec4i>& hierarchy)
{
    // cv::Mat image;
    // copyMakeBorder(_image, image, 1, 1, 1, 1, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0) );
    biMakeBorder(image, 0);

    cv::MemStorage storage(cvCreateMemStorage());
    CvSeq* _ccontours = 0;
  
    int total = findContoursImpl(image, storage, &_ccontours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);        
    _contours.resize(total);
    hierarchy.resize(total);

    cv::Seq<CvSeq*> contours_seq = treeToNodeSeq( _ccontours, sizeof(CvSeq), storage );

    //printf("count:%d, total:%d\n", count, total);

    for(int i = 0; i < total; i++)
    {
        CvSeq* c = contours_seq[i];
        ((struct Contour*)c)->color = i;
        _contours[i].resize( c->total );
 
        CvtSeqToVector(c, _contours[i]);

        int h_next = c->h_next ? ((struct Contour*)c->h_next)->color : -1;
        int h_prev = c->h_prev ? ((struct Contour*)c->h_prev)->color : -1;
        int v_next = c->v_next ? ((struct Contour*)c->v_next)->color : -1;
        int v_prev = c->v_prev ? ((struct Contour*)c->v_prev)->color : -1;
        
        hierarchy[i] = cv::Vec4i(h_next, h_prev, v_next, v_prev);
    }
}
