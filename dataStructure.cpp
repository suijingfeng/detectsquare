

// Insert contour into tree given certain parent sequence.
// If parent is equal to frame (the most external contour),
// then added contour will have null pointer to parent:
void cvInsertNodeIntoTree( void* _node, void* _parent, void* _frame )
{
    CvTreeNode* node = (CvTreeNode*)_node;
    CvTreeNode* parent = (CvTreeNode*)_parent;

    if( !node || !parent )
        CV_Error( CV_StsNullPtr, "" );

    node->v_prev = (_parent != _frame ? parent : 0);
    node->h_next = parent->v_next;

    assert( parent->v_next != node );

    if( parent->v_next )
        parent->v_next->h_prev = node;
    parent->v_next = node;
}


// Remove contour from tree, together with the contour's children:
void cvRemoveNodeFromTree( void* _node, void* _frame )
{
    CvTreeNode* node = (CvTreeNode*)_node;
    CvTreeNode* frame = (CvTreeNode*)_frame;

    if( !node )
        CV_Error( CV_StsNullPtr, "" );

    if( node == frame )
        CV_Error( CV_StsBadArg, "frame node could not be deleted" );

    if( node->h_next )
        node->h_next->h_prev = node->h_prev;

    if( node->h_prev )
        node->h_prev->h_next = node->h_next;
    else
    {
        CvTreeNode* parent = node->v_prev;
        if( !parent )
            parent = frame;

        if( parent )
        {
            assert( parent->v_next == node );
            parent->v_next = node->h_next;
        }
    }
}


void cvInitTreeNodeIterator( CvTreeNodeIterator* treeIterator, const void* first, int max_level )
{
    if( !treeIterator || !first )
        CV_Error( CV_StsNullPtr, "" );

    if( max_level < 0 )
        CV_Error( CV_StsOutOfRange, "" );

    treeIterator->node = (void*)first;
    treeIterator->level = 0;
    treeIterator->max_level = max_level;
}


void* cvNextTreeNode( CvTreeNodeIterator* treeIterator )
{
    CvTreeNode* prevNode = 0;
    CvTreeNode* node;
    int level;

    if( !treeIterator )
        CV_Error( CV_StsNullPtr, "NULL iterator pointer" );

    prevNode = node = (CvTreeNode*)treeIterator->node;
    level = treeIterator->level;

    if( node )
    {
        if( node->v_next && level+1 < treeIterator->max_level )
        {
            node = node->v_next;
            level++;
        }
        else
        {
            while( node->h_next == 0 )
            {
                node = node->v_prev;
                if( --level < 0 )
                {
                    node = 0;
                    break;
                }
            }
            node = node && treeIterator->max_level != 0 ? node->h_next : 0;
        }
    }

    treeIterator->node = node;
    treeIterator->level = level;
    return prevNode;
}


void* cvPrevTreeNode( CvTreeNodeIterator* treeIterator )
{
    CvTreeNode* prevNode = 0;
    CvTreeNode* node;
    int level;

    if( !treeIterator )
        CV_Error( CV_StsNullPtr, "" );

    prevNode = node = (CvTreeNode*)treeIterator->node;
    level = treeIterator->level;

    if( node )
    {
        if( !node->h_prev )
        {
            node = node->v_prev;
            if( --level < 0 )
                node = 0;
        }
        else
        {
            node = node->h_prev;

            while( node->v_next && level < treeIterator->max_level )
            {
                node = node->v_next;
                level++;

                while( node->h_next )
                    node = node->h_next;
            }
        }
    }

    treeIterator->node = node;
    treeIterator->level = level;
    return prevNode;
}

inline int cvAlignLeft( int size, int align )
{
    return size & -align;
}


/* Moves stack pointer to next block.
   If no blocks, allocate new one and link it to the storage: */
static void icvGoNextMemBlock( CvMemStorage * storage )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    if( !storage->top || !storage->top->next )
    {
        CvMemBlock *block;

        if( !(storage->parent) )
        {
            block = (CvMemBlock *)cvAlloc( storage->block_size );
        }
        else
        {
            CvMemStorage *parent = storage->parent;
            CvMemStoragePos parent_pos;

            cvSaveMemStoragePos( parent, &parent_pos );
            icvGoNextMemBlock( parent );

            block = parent->top;
            cvRestoreMemStoragePos( parent, &parent_pos );

            if( block == parent->top )  /* the single allocated block */
            {
                assert( parent->bottom == block );
                parent->top = parent->bottom = 0;
                parent->free_space = 0;
            }
            else
            {
                /* cut the block from the parent's list of blocks */
                parent->top->next = block->next;
                if( block->next )
                    block->next->prev = parent->top;
            }
        }

        /* link block */
        block->next = 0;
        block->prev = storage->top;

        if( storage->top )
            storage->top->next = block;
        else
            storage->top = storage->bottom = block;
    }

    if( storage->top->next )
        storage->top = storage->top->next;
    storage->free_space = storage->block_size - sizeof(CvMemBlock);
    assert( storage->free_space % CV_STRUCT_ALIGN == 0 );
}


/* Allocate continuous buffer of the specified size in the storage: */
void* cvMemStorageAlloc( CvMemStorage* storage, size_t size )
{
    schar *ptr = 0;
    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    if( size > INT_MAX )
        CV_Error( CV_StsOutOfRange, "Too large memory block is requested" );

    assert( storage->free_space % CV_STRUCT_ALIGN == 0 );

    if( (size_t)storage->free_space < size )
    {
        size_t max_free_space = cvAlignLeft(storage->block_size - sizeof(CvMemBlock), CV_STRUCT_ALIGN);
        if( max_free_space < size )
            CV_Error( CV_StsOutOfRange, "requested size is negative or too big" );

        icvGoNextMemBlock( storage );
    }

    ptr = ICV_FREE_PTR(storage);
    assert( (size_t)ptr % CV_STRUCT_ALIGN == 0 );
    storage->free_space = cvAlignLeft(storage->free_space - (int)size, CV_STRUCT_ALIGN );

    return ptr;
}

/****************************************************************************************\
*                               Sequence implementation                                  *
\****************************************************************************************/

/* Create empty sequence: */
CV_IMPL CvSeq* cvCreateSeq(int seq_flags, size_t header_size, size_t elem_size, CvMemStorage* storage)
{
    CvSeq *seq = 0;

    if( !storage )
        CV_Error( CV_StsNullPtr, "" );
    if( header_size < sizeof( CvSeq ) || elem_size <= 0 )
        CV_Error( CV_StsBadSize, "" );

    /* allocate sequence header */
    seq = (CvSeq*)cvMemStorageAlloc( storage, header_size );
    memset( seq, 0, header_size );

    seq->header_size = (int)header_size;
    seq->flags = (seq_flags & ~CV_MAGIC_MASK) | CV_SEQ_MAGIC_VAL;
    {
        int elemtype = CV_MAT_TYPE(seq_flags);
        int typesize = CV_ELEM_SIZE(elemtype);

        if( elemtype != CV_SEQ_ELTYPE_GENERIC && elemtype != CV_USRTYPE1 &&
            typesize != 0 && typesize != (int)elem_size )
            CV_Error( CV_StsBadSize,
            "Specified element size doesn't match to the size of the specified element type "
            "(try to use 0 for element type)" );
    }
    seq->elem_size = (int)elem_size;
    seq->storage = storage;

    cvSetSeqBlockSize( seq, (int)((1 << 10)/elem_size) );

    return seq;
}


/* adjusts <delta_elems> field of sequence. It determines how much the sequence
   grows if there are no free space inside the sequence buffers */
CV_IMPL void cvSetSeqBlockSize( CvSeq *seq, int delta_elements )
{
    int elem_size;
    int useful_block_size;

    if( !seq || !seq->storage )
        CV_Error( CV_StsNullPtr, "" );
    if( delta_elements < 0 )
        CV_Error( CV_StsOutOfRange, "" );

    useful_block_size = cvAlignLeft(seq->storage->block_size - sizeof(CvMemBlock) - sizeof(CvSeqBlock), CV_STRUCT_ALIGN);
    elem_size = seq->elem_size;

    if( delta_elements == 0 )
    {
        delta_elements = (1 << 10) / elem_size;
        delta_elements = MAX( delta_elements, 1 );
    }
    if( delta_elements * elem_size > useful_block_size )
    {
        delta_elements = useful_block_size / elem_size;
        if( delta_elements == 0 )
            CV_Error( CV_StsOutOfRange, "Storage block size is too small "
                                        "to fit the sequence elements" );
    }

    seq->delta_elems = delta_elements;
}


/* Restore memory storage position: */
CV_IMPL void cvRestoreMemStoragePos(CvMemStorage * storage, CvMemStoragePos * pos)
{
    if( !storage || !pos )
        CV_Error( CV_StsNullPtr, "" );
    if( pos->free_space > storage->block_size )
        CV_Error( CV_StsBadSize, "" );

    /*
    // this breaks icvGoNextMemBlock, so comment it off for now
    if( storage->parent && (!pos->top || pos->top->next) )
    {
        CvMemBlock* save_bottom;
        if( !pos->top )
            save_bottom = 0;
        else
        {
            save_bottom = storage->bottom;
            storage->bottom = pos->top->next;
            pos->top->next = 0;
            storage->bottom->prev = 0;
        }
        icvDestroyMemStorage( storage );
        storage->bottom = save_bottom;
    }*/

    storage->top = pos->top;
    storage->free_space = pos->free_space;

    if( !storage->top )
    {
        storage->top = storage->bottom;
        storage->free_space = storage->top ? storage->block_size - sizeof(CvMemBlock) : 0;
    }
}


/* Clears memory storage (return blocks to the parent, if any): */
CV_IMPL void cvClearMemStorage( CvMemStorage * storage )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    if( storage->parent )
        icvDestroyMemStorage( storage );
    else
    {
        storage->top = storage->bottom;
        storage->free_space = storage->bottom ? storage->block_size - sizeof(CvMemBlock) : 0;
    }
}



/* The function allocates space for at least one more sequence element.
   If there are free sequence blocks (seq->free_blocks != 0)
   they are reused, otherwise the space is allocated in the storage: */
static void icvGrowSeq( CvSeq *seq, int in_front_of )
{
    CvSeqBlock *block;

    if( !seq )
        CV_Error( CV_StsNullPtr, "" );
    block = seq->free_blocks;

    if( !block )
    {
        int elem_size = seq->elem_size;
        int delta_elems = seq->delta_elems;
        CvMemStorage *storage = seq->storage;

        if( seq->total >= delta_elems*4 )
            cvSetSeqBlockSize( seq, delta_elems*2 );

        if( !storage )
            CV_Error( CV_StsNullPtr, "The sequence has NULL storage pointer" );

        /* If there is a free space just after last allocated block
           and it is big enough then enlarge the last block.
           This can happen only if the new block is added to the end of sequence: */
        if( (size_t)(ICV_FREE_PTR(storage) - seq->block_max) < CV_STRUCT_ALIGN &&
            storage->free_space >= seq->elem_size && !in_front_of )
        {
            int delta = storage->free_space / elem_size;

            delta = MIN( delta, delta_elems ) * elem_size;
            seq->block_max += delta;
            storage->free_space = cvAlignLeft((int)(((schar*)storage->top + storage->block_size) -
                                              seq->block_max), CV_STRUCT_ALIGN );
            return;
        }
        else
        {
            int delta = elem_size * delta_elems + ICV_ALIGNED_SEQ_BLOCK_SIZE;

            /* Try to allocate <delta_elements> elements: */
            if( storage->free_space < delta )
            {
                int small_block_size = MAX(1, delta_elems/3)*elem_size + ICV_ALIGNED_SEQ_BLOCK_SIZE;
                /* try to allocate smaller part */
                if( storage->free_space >= small_block_size + CV_STRUCT_ALIGN )
                {
                    delta = (storage->free_space - ICV_ALIGNED_SEQ_BLOCK_SIZE)/seq->elem_size;
                    delta = delta*seq->elem_size + ICV_ALIGNED_SEQ_BLOCK_SIZE;
                }
                else
                {
                    icvGoNextMemBlock( storage );
                    assert( storage->free_space >= delta );
                }
            }

            block = (CvSeqBlock*)cvMemStorageAlloc( storage, delta );
            block->data = (schar*)cvAlignPtr( block + 1, CV_STRUCT_ALIGN );
            block->count = delta - ICV_ALIGNED_SEQ_BLOCK_SIZE;
            block->prev = block->next = 0;
        }
    }
    else
    {
        seq->free_blocks = block->next;
    }

    if( !(seq->first) )
    {
        seq->first = block;
        block->prev = block->next = block;
    }
    else
    {
        block->prev = seq->first->prev;
        block->next = seq->first;
        block->prev->next = block->next->prev = block;
    }

    /* For free blocks the <count> field means
     * total number of bytes in the block.
     *
     * For used blocks it means current number
     * of sequence elements in the block:
     */
    assert( block->count % seq->elem_size == 0 && block->count > 0 );

    if( !in_front_of )
    {
        seq->ptr = block->data;
        seq->block_max = block->data + block->count;
        block->start_index = block == block->prev ? 0 :
            block->prev->start_index + block->prev->count;
    }
    else
    {
        int delta = block->count / seq->elem_size;
        block->data += block->count;

        if( block != block->prev )
        {
            assert( seq->first->start_index == 0 );
            seq->first = block;
        }
        else
        {
            seq->block_max = seq->ptr = block->data;
        }

        block->start_index = 0;

        for( ;; )
        {
            block->start_index += delta;
            block = block->next;
            if( block == seq->first )
                break;
        }
    }

    block->count = 0;
}


/* Push element onto the sequence: */
CV_IMPL schar* cvSeqPush( CvSeq *seq, const void *element )
{
    if( !seq )
        CV_Error( CV_StsNullPtr, "" );

    size_t elem_size = seq->elem_size;
    schar *ptr = seq->ptr;

    if( ptr >= seq->block_max )
    {
        icvGrowSeq( seq, 0 );

        ptr = seq->ptr;
        assert( ptr + elem_size <= seq->block_max /*&& ptr == seq->block_min */  );
    }

    if( element )
        memcpy( ptr, element, elem_size );
    seq->first->prev->count++;
    seq->total++;
    seq->ptr = ptr + elem_size;

    return ptr;
}


