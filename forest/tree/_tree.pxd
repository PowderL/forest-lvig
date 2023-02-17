import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer
ctypedef np.npy_int64 INT64_t
#ctypedef np.npy_uint8 BOOL_t

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    #DOUBLE_t local_impurity
    #DOUBLE_t local_y_sum
    #DOUBLE_t local_squared_y_sum
    #SIZE_t local_case_num

cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.
    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t n_features
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes

    # Methods
    #cpdef SIZE_t _rebuild_tree(self, object, tree_attrs)
    cpdef SIZE_t _rebuild_node(self, SIZE_t node_id, bint is_leaf, SIZE_t left_child, 
          SIZE_t right_child, double threshold, SIZE_t feature)
              
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1
    cpdef DOUBLE_t[:,:] local_variable_importance_g(self, object X, object y, object grouping)
    cdef inline DOUBLE_t[:,:] _parse(self, object X, np.ndarray y, np.ndarray grouping)