#ifndef DATASTRUCTURE_H_
#define DATASTRUCTURE_H_


#define CV_TREE_NODE_FIELDS(node_type)                               \



typedef struct CvSeqBlock
{
    struct CvSeqBlock* prev; /**< Previous sequence block.                   */
    struct CvSeqBlock* next; /**< Next sequence block.                       */
    int    start_index;       /**< Index of the first element in the block +  */
                              /**< sequence->first->start_index.              */
    int    count;             /**< Number of elements in the block.           */
    schar* data;              /**< Pointer to the first element of the block. */
}CvSeqBlock;


/**
   Read/Write sequence.
   Elements can be dynamically inserted to or deleted from the sequence.
*/
typedef struct CvSeq
{
    int       flags;            /**< Miscellaneous flags.     */
    int       header_size;      /**< Size of sequence header. */
    struct    CvSeq* h_prev;    /**< Previous sequence.       */
    struct    CvSeq* h_next;    /**< Next sequence.           */
    struct    CvSeq* v_prev;    /**< 2nd previous sequence.   */
    struct    CvSeq* v_next     /**< 2nd next sequence.       */
                                           
    int       total;          /**< Total number of elements.            */
    int       elem_size;      /**< Size of sequence element in bytes.   */
    schar*    block_max;      /**< Maximal bound of the last block.     */
    schar*    ptr;            /**< Current write pointer.               */
    int       delta_elems;    /**< Grow seq this many at a time.        */
    CvMemStorage* storage;    /**< Where the seq is stored.             */
    CvSeqBlock* free_blocks;  /**< Free blocks list.                    */
    CvSeqBlock* first;        /**< Pointer to the first sequence block. */
}CvSeq;



typedef struct CvTreeNode
{
    int       flags;         /* micsellaneous flags */
    int       header_size;   /* size of sequence header */
    struct    CvTreeNode* h_prev; /* previous sequence */
    struct    CvTreeNode* h_next; /* next sequence */
    struct    CvTreeNode* v_prev; /* 2nd previous sequence */
    struct    CvTreeNode* v_next; /* 2nd next sequence */
}
CvTreeNode;


typedef struct CvSeqReader
{
    int          header_size;                           
    CvSeq*       seq;        /**< sequence, beign read */
    CvSeqBlock*  block;      /**< current block */
    schar*       ptr;        /**< pointer to element be read next */
    schar*       block_min;  /**< pointer to the beginning of block */
    schar*       block_max;  /**< pointer to the end of block */
    int          delta_index;/**< = seq->first->start_index   */
    schar*       prev_elem;  /**< pointer to previous element */
}CvSeqReader;


/******************* Iteration through the sequence tree *****************/
typedef struct CvTreeNodeIterator
{
    const void* node;
    int level;
    int max_level;
}
CvTreeNodeIterator;

CVAPI(void) cvInitTreeNodeIterator( CvTreeNodeIterator* tree_iterator, const void* first, int max_level );
CVAPI(void*) cvNextTreeNode( CvTreeNodeIterator* tree_iterator );
CVAPI(void*) cvPrevTreeNode( CvTreeNodeIterator* tree_iterator );

/** Inserts sequence into tree with specified "parent" sequence.
   If parent is equal to frame (e.g. the most external contour),
   then added contour will have null pointer to parent. */
CVAPI(void) cvInsertNodeIntoTree( void* node, void* parent, void* frame );

/** Removes contour from tree (together with the contour children). */
CVAPI(void) cvRemoveNodeFromTree( void* node, void* frame );

/** Gathers pointers to all the sequences, accessible from the `first`, to the single sequence */
CVAPI(CvSeq*) cvTreeToNodeSeq( const void* first, int header_size, CvMemStorage* storage );

#endif
