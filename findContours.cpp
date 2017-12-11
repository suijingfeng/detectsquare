#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#include "findContours.h"
#include "threshold.h"

using namespace std;


struct CvLinkedRunPoint
{
    struct CvLinkedRunPoint* link;
    struct CvLinkedRunPoint* next;
    CvPoint pt;
};

#define CV_GET_WRITTEN_ELEM( writer ) ((writer).ptr - (writer).seq->elem_size)
#define ICV_SINGLE              0
#define ICV_CONNECTING_ABOVE    1
#define ICV_CONNECTING_BELOW    -1

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


inline int findStartContourPoint(uchar *src_data, CvSize img_size, int j, bool haveSIMD)
{
#if CV_SSE2
    if (haveSIMD)
    {
        __m128i v_zero = _mm_setzero_si128();
        int v_size = img_size.width - 32;

        for (; j <= v_size; j += 32) {
            __m128i v_p1 = _mm_loadu_si128((const __m128i*)(src_data + j));
            __m128i v_p2 = _mm_loadu_si128((const __m128i*)(src_data + j + 16));

            __m128i v_cmp1 = _mm_cmpeq_epi8(v_p1, v_zero);
            __m128i v_cmp2 = _mm_cmpeq_epi8(v_p2, v_zero);

            unsigned int mask1 = _mm_movemask_epi8(v_cmp1);
            unsigned int mask2 = _mm_movemask_epi8(v_cmp2);

            mask1 ^= 0x0000ffff;
            mask2 ^= 0x0000ffff;

            if (mask1) {
                j += trailingZeros(mask1);
                return j;
            }

            if (mask2) {
                j += trailingZeros(mask2 << 16);
                return j;
            }
        }

        if (j <= img_size.width - 16)
        {
            __m128i v_p = _mm_loadu_si128((const __m128i*)(src_data + j));

            unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v_p, v_zero)) ^ 0x0000ffff;

            if (mask) {
                j += trailingZeros(mask);
                return j;
            }
            j += 16;
        }
    }
#else
    CV_UNUSED(haveSIMD);
#endif
    for(; j < img_size.width && !src_data[j]; ++j)
        ;
    return j;
}


inline int findEndContourPoint(unsigned char *src_data, CvSize img_size, int j, bool haveSIMD)
{
#if CV_SSE2
    if (j < img_size.width && !src_data[j])
        return j;
    else if (haveSIMD)
    {
        __m128i v_zero = _mm_setzero_si128();
        int v_size = img_size.width - 32;

        for(; j <= v_size; j += 32)
        {
            __m128i v_p1 = _mm_loadu_si128((const __m128i*)(src_data + j));
            __m128i v_p2 = _mm_loadu_si128((const __m128i*)(src_data + j + 16));

            __m128i v_cmp1 = _mm_cmpeq_epi8(v_p1, v_zero);
            __m128i v_cmp2 = _mm_cmpeq_epi8(v_p2, v_zero);

            unsigned int mask1 = _mm_movemask_epi8(v_cmp1);
            unsigned int mask2 = _mm_movemask_epi8(v_cmp2);

            if(mask1)
            {
                j += trailingZeros(mask1);
                return j;
            }

            if(mask2)
            {
                j += trailingZeros(mask2 << 16);
                return j;
            }
        }

        if(j <= img_size.width - 16)
        {
            __m128i v_p = _mm_loadu_si128((const __m128i*)(src_data + j));

            unsigned int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(v_p, v_zero));

            if (mask)
            {
                j += trailingZeros(mask);
                return j;
            }
            j += 16;
        }
    }
#else
    CV_UNUSED(haveSIMD);
#endif
    for(; j < img_size.width && src_data[j]; ++j)
        ;

    return j;
}




CvSeq* TreeToNodeSeq(const void* first, int header_size, CvMemStorage* storage )
{
    CvSeq* allseq = 0;
    CvTreeNodeIterator iterator;

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    allseq = cvCreateSeq( 0, header_size, sizeof(first), storage );

    if( first )
    {
        cvInitTreeNodeIterator( &iterator, first, INT_MAX );

        for(;;)
        {
            void* node = cvNextTreeNode( &iterator );
            if( !node )
                break;
            cvSeqPush( allseq, &node );
        }
    }

    return allseq;
}


/* Copy all sequence elements into single std::vector */
void CvtSeqToVector(const CvSeq *seq, std::vector<cv::Point> &p)
{
    CvSeqReader reader;

    int total = seq->total; // *elem_size();
    if( total == 0 )
        return;
    p.resize(total);
    cvStartReadSeq( seq, &reader, 0 );
    cv::Point temp;
    for (int i = 0; i<seq->total; i++)
    {
        CV_READ_SEQ_ELEM(temp, reader);
        p[i] = temp;
    }
}


static void EndProcessContour( ContourScanner scanner )
{
    _CvContourInfo *l_cinfo = scanner->l_cinfo;

    if( l_cinfo )
    {
        if( scanner->subst_flag )
        {
            CvMemStoragePos temp;

            cvSaveMemStoragePos( scanner->storage2, &temp );

            if( temp.top == scanner->backup_pos2.top && temp.free_space == scanner->backup_pos2.free_space )
            {
                cvRestoreMemStoragePos( scanner->storage2, &scanner->backup_pos );
            }
            scanner->subst_flag = 0;
        }

        if( l_cinfo->contour )
        {
            cvInsertNodeIntoTree( l_cinfo->contour, l_cinfo->parent->contour, &(scanner->frame) );
        }
        scanner->l_cinfo = 0;
    }
}


static int TraceContour_32s( int *ptr, int step, int *stop_ptr, int is_hole )
{
    int deltas[16];
    int *i0 = ptr, *i1, *i3, *i4;
    int s, s_end;
    const int   right_flag = INT_MIN;
    const int   new_flag = (int)((unsigned)INT_MIN >> 1);
    const int   value_mask = ~(right_flag | new_flag);
    const int   ccomp_val = *i0 & value_mask;

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    s_end = s = is_hole ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( (*i1 & value_mask) != ccomp_val && s != s_end );

    i3 = i0;

    /* check single pixel domain */
    if( s != s_end )
    {
        /* follow border */
        for( ;; )
        {
            s_end = s;

            for( ;; )
            {
                i4 = i3 + deltas[++s];
                if( (*i4 & value_mask) == ccomp_val )
                    break;
            }

            if( i3 == stop_ptr || (i4 == i0 && i3 == i1) )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }
    return i3 == stop_ptr;
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
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    assert( (*i0 & -2) != 0 );

    s_end = s = is_hole ? 0 : 4;

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



static const CvPoint icvCodeDeltas[8] =
    { CvPoint(1, 0), CvPoint(1, -1), CvPoint(0, -1), CvPoint(-1, -1), CvPoint(-1, 0), CvPoint(-1, 1), CvPoint(0, 1), CvPoint(1, 1) };

/*
    marks domain border with +/-<constant> and stores the contour into CvSeq.
        method:
            <0  - chain
            ==0 - direct
            >0  - simple approximation
*/
static void FetchContour( schar *ptr, int step, CvPoint pt, CvSeq* contour, int _method )
{
    const schar     nbd = 2;
    int             deltas[16];
    CvSeqWriter     writer;
    schar           *i0 = ptr, *i1, *i3, *i4 = 0;
    int             prev_s = -1, s, s_end;
    int             method = _method - 1;

    assert( (unsigned) _method <= CV_CHAIN_APPROX_SIMPLE );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
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
            CV_WRITE_SEQ_ELEM( pt, writer );
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

                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else
            {
                if( s != prev_s || method == 0 )
                {
                    CV_WRITE_SEQ_ELEM( pt, writer );
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
    schar        *i0 = ptr, *i1, *i3, *i4;
    CvRect      rect;
    int         prev_s = -1, s, s_end;
    int         method = _method - 1;

    assert( (unsigned) _method <= CV_CHAIN_APPROX_SIMPLE );
    assert( 1 < nbd && nbd < 128 );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
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
            CV_WRITE_SEQ_ELEM( pt, writer );
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
                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else if( s != prev_s || method == 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
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

    if( _rect )  *_rect = rect;
}


static void icvFetchContourEx_32s( int* ptr, int step, CvPoint pt, CvSeq* contour, int _method, CvRect* _rect )
{
    int         deltas[16];
    CvSeqWriter writer;
    int        *i0 = ptr, *i1, *i3, *i4;
    CvRect      rect;
    int         prev_s = -1, s, s_end;
    int         method = _method - 1;
    const int   right_flag = INT_MIN;
    const int   new_flag = (int)((unsigned)INT_MIN >> 1);
    const int   value_mask = ~(right_flag | new_flag);
    const int   ccomp_val = *i0 & value_mask;
    const int   nbd0 = ccomp_val | new_flag;
    const int   nbd1 = nbd0 | right_flag;

    assert( (unsigned) _method <= CV_CHAIN_APPROX_SIMPLE );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
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
    while( (*i1 & value_mask) != ccomp_val && s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = nbd1;
        if( method >= 0 )
        {
            CV_WRITE_SEQ_ELEM( pt, writer );
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
                if( (*i4 & value_mask) == ccomp_val )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = nbd1;
            }
            else if( *i3 == ccomp_val )
            {
                *i3 = nbd0;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;
                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else if( s != prev_s || method == 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
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


typedef struct _CvPtInfo
{
    CvPoint pt;
    int k;                      /* support region */
    int s;                      /* curvature value */
    struct _CvPtInfo *next;
}
_CvPtInfo;


CvSeq* FindNextContour( ContourScanner scanner )
{
    if( !scanner )
        CV_Error( CV_StsNullPtr, "" );

#if CV_SSE2
    bool haveSIMD = cv::checkHardwareSupport(CPU_SSE2);
#endif

    CV_Assert(scanner->img_step >= 0);

    EndProcessContour( scanner );

    /* initialize local state */
    schar* img0 = scanner->img0;
    schar* img = scanner->img;
    int step = scanner->img_step;
    int step_i = step / sizeof(int);
    int x = scanner->pt.x;
    int y = scanner->pt.y;
    int width = scanner->img_size.width;
    int height = scanner->img_size.height;
    int mode = scanner->mode;
    CvPoint lnbd = scanner->lnbd;
    int nbd = scanner->nbd;
    int prev = img[x - 1];
    int new_mask = -2;

    for( ; y < height; y++, img += step )
    {
        int* img0_i = 0;
        int* img_i = 0;
        int p = 0;

        for( ; x < width; x++ )
        {
            if( img_i )
            {
                for( ; x < width && ((p = img_i[x]) == prev || (p & ~new_mask) == (prev & ~new_mask)); x++ )
                    prev = p;
            }
            else
            {
#if CV_SSE2
                if ((p = img[x]) != prev) {
                    goto _next_contour;
                } else if (haveSIMD) {

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
                for( ; x < width && (p = img[x]) == prev; x++ )
                    ;
            }

            if( x >= width )
                break;
#if CV_SSE2
        _next_contour:
#endif
            {
                _CvContourInfo *par_info = 0;
                _CvContourInfo *l_cinfo = 0;
                CvSeq *seq = 0;
                int is_hole = 0;
                CvPoint origin;

                /* if not external contour */
                if( (!img_i && !(prev == 0 && p == 1)) ||
                    (img_i && !(((prev & new_mask) != 0 || prev == 0) && (p & new_mask) == 0)) )
                {
                    /* check hole */
                    if( (!img_i && (p != 0 || prev < 1)) ||
                        (img_i && ((prev & new_mask) != 0 || (p & new_mask) != 0)))
                        goto resume_scan;

                    if( prev & new_mask )
                    {
                        lnbd.x = x - 1;
                    }
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
                    int lval = (img0_i ? img0_i[lnbd.y * static_cast<size_t>(step_i) + lnbd.x] :
                        (int)img0[lnbd.y * static_cast<size_t>(step) + lnbd.x]) & 0x7f;
                    _CvContourInfo *cur = scanner->cinfo_table[lval];

                    /* find the first bounding contour */
                    while( cur )
                    {
                        if( (unsigned) (lnbd.x - cur->rect.x) < (unsigned) cur->rect.width &&
                            (unsigned) (lnbd.y - cur->rect.y) < (unsigned) cur->rect.height )
                        {
                            if( par_info )
                            {
                                if( (img0_i && TraceContour_32s( img0_i + par_info->origin.y * static_cast<size_t>(step_i) +
                                                          par_info->origin.x, step_i, img_i + lnbd.x, par_info->is_hole ) > 0) ||
                                    (!img0_i && TraceContour( img0 + par_info->origin.y * static_cast<size_t>(step) +
                                                      par_info->origin.x, step, img + lnbd.x, par_info->is_hole ) > 0) )
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

                seq = cvCreateSeq( scanner->seq_type1, scanner->header_size1,
                                   scanner->elem_size1, scanner->storage1 );
                seq->flags |= is_hole ? CV_SEQ_FLAG_HOLE : 0;

                /* initialize header */
                if( mode <= 1 )
                {
                    l_cinfo = &(scanner->cinfo_temp);
                    FetchContour( img + x - is_hole, step, cvPoint( origin.x + scanner->offset.x, origin.y + scanner->offset.y), seq, scanner->approx_method1 );
                }
                else
                {
                    union { _CvContourInfo* ci; CvSetElem* se; } v;
                    v.ci = l_cinfo;
                    cvSetAdd( scanner->cinfo_set, 0, &v.se );
                    l_cinfo = v.ci;
                    int lval;

                    if( img_i )
                    {
                        lval = img_i[x - is_hole] & 127;
                        icvFetchContourEx_32s(img_i + x - is_hole, step_i, cvPoint( origin.x + scanner->offset.x, origin.y + scanner->offset.y),
                                              seq, scanner->approx_method1, &(l_cinfo->rect) );
                    }
                    else
                    {
                        lval = nbd;
                        // change nbd
                        nbd = (nbd + 1) & 127;
                        nbd += nbd == 0 ? 3 : 0;
                        icvFetchContourEx( img + x - is_hole, step,
                                           cvPoint( origin.x + scanner->offset.x,
                                                    origin.y + scanner->offset.y),
                                           seq, scanner->approx_method1,
                                           lval, &(l_cinfo->rect) );
                    }
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
                    {
                        cvRestoreMemStoragePos( scanner->storage1, &(scanner->backup_pos) );
                    }
                    else
                    {
                        cvClearMemStorage( scanner->storage1 );
                    }
                    p = img[x];
                    goto resume_scan;
                }

                cvSaveMemStoragePos( scanner->storage2, &(scanner->backup_pos2) );
                scanner->l_cinfo = l_cinfo;
                scanner->pt.x = !img_i ? x + 1 : x + 1 - is_hole;
                scanner->pt.y = y;
                scanner->lnbd = lnbd;
                scanner->img = (schar *) img;
                scanner->nbd = nbd;
                return l_cinfo->contour;

            resume_scan:

                prev = p;
                /* update lnbd */
                if( prev & -2 )
                {
                    lnbd.x = x;
                }
            }                   /* end of prev != p */
        }                       /* end of loop on x */

        lnbd.x = 0;
        lnbd.y = y + 1;
        x = 1;
        prev = 0;
    }                           /* end of loop on y */

    return 0;
}


static void icvEndProcessContour( ContourScanner scanner )
{
    _CvContourInfo *l_cinfo = scanner->l_cinfo;

    if( l_cinfo )
    {
        if( scanner->subst_flag )
        {
            CvMemStoragePos temp;

            cvSaveMemStoragePos( scanner->storage2, &temp );

            if( temp.top == scanner->backup_pos2.top &&
                temp.free_space == scanner->backup_pos2.free_space )
            {
                cvRestoreMemStoragePos( scanner->storage2, &scanner->backup_pos );
            }
            scanner->subst_flag = 0;
        }

        if( l_cinfo->contour )
        {
            cvInsertNodeIntoTree( l_cinfo->contour, l_cinfo->parent->contour,
                                  &(scanner->frame) );
        }
        scanner->l_cinfo = 0;
    }
}

/* The function add to tree the last retrieved/substituted contour,
   releases temp_storage, restores state of dst_storage (if needed),
   and returns pointer to root of the contour tree */

CvSeq *cvEndFindContours(ContourScanner * _scanner )
{
    ContourScanner scanner;
    CvSeq *first = 0;

    if( !_scanner )
        CV_Error( CV_StsNullPtr, "" );
    scanner = *_scanner;

    if( scanner )
    {
        icvEndProcessContour( scanner );

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
static ContourScanner startFindContours_Impl( cv::Mat& img, CvMemStorage* storage, int mode, int method, CvPoint offset)
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    ContourScanner scanner = (ContourScanner)cvAlloc( sizeof( *scanner ));
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
    scanner->subst_flag = 0;

    scanner->frame.flags = CV_SEQ_FLAG_HOLE;

    scanner->approx_method2 = scanner->approx_method1 = method;

    scanner->seq_type1 = scanner->seq_type2 = CV_SEQ_POLYGON;
    scanner->header_size1 = scanner->header_size2 = sizeof( struct Contour );
    scanner->elem_size1 =  scanner->elem_size2 = sizeof( CvPoint );

    cvSaveMemStoragePos( storage, &(scanner->initial_pos) );

    if( method > CV_CHAIN_APPROX_SIMPLE ) // CV_CHAIN_APPROX_SIMPLE = 2
    {
        scanner->storage1 = cvCreateChildMemStorage( scanner->storage2 );
    }

    if( mode > CV_RETR_LIST ) // CV_RETR_LIST =1, CV_RETR_CCOMP =2 
    {
        scanner->cinfo_storage = cvCreateChildMemStorage( scanner->storage2 );
        scanner->cinfo_set = cvCreateSet( 0, sizeof( CvSet ), sizeof( _CvContourInfo ), scanner->cinfo_storage );
    }

    // converts all pixels to 0 or 1
    threshold(img, 0, 1 );
    return scanner;
}


static int findContoursImpl( cv::Mat& image, CvMemStorage* storage, CvSeq** firstContour, int mode, int method, CvPoint offset)
{
    ContourScanner scanner = 0;
    CvSeq *contour = 0;
    int count = -1;

    if( !firstContour )
        CV_Error( CV_StsNullPtr, "NULL double CvSeq pointer" );

    *firstContour = 0;
    
    try
    {
        scanner = startFindContours_Impl(image, storage, mode, method, offset);

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
            cvEndFindContours(&scanner);
        throw;
    }

    *firstContour = cvEndFindContours( &scanner );
    
    return count;
}


void findContours(cv::Mat& image, std::vector< std::vector<cv::Point> >& _contours, std::vector<cv::Vec4i>& hierarchy)
{
    // cv::Mat image;
    // copyMakeBorder(_image, image, 1, 1, 1, 1, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0) );
    biMakeBorder(image, 0);

    cv::MemStorage storage(cvCreateMemStorage());
    CvSeq* _ccontours = 0;
  
    findContoursImpl(image, storage, &_ccontours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    cv::Seq<CvSeq*> contours_seq( TreeToNodeSeq( _ccontours, sizeof(CvSeq), storage ));
        
    int total = contours_seq.size();
    _contours.resize(total);
    hierarchy.resize(total);

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
