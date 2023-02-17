from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport safe_realloc
## at first we need to rebuild the decesion tree, and utilizing the re-build tree to compute importance
TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

cdef class Tree:
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    def __cinit__(self, int n_features, int max_depth):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        # Inner structures
        self.max_depth = max_depth
        self.node_count = 0
        cdef capacity = 2**(max_depth + 1) - 1
        self.nodes = NULL
        self._resize(capacity)
        
    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.nodes)

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
       # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0
    ## 遍历tree，将参数输入到nodes中
    cpdef SIZE_t _rebuild_node(self, SIZE_t node_id, bint is_leaf, SIZE_t left_child, 
          SIZE_t right_child, double threshold, SIZE_t feature):
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        ## 正在考虑要去除这个
        cdef Node* node = &self.nodes[node_id]
        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX
        ## 由于是重新构建树，考虑不适用impurity了
        #node.impurity = impurity
        ## 在这里recursively rebuild nodes
        ## rebuild children        
        if not is_leaf:
            node.left_child = left_child
            node.right_child = right_child
            node.feature = feature
            node.threshold = threshold
        else:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED
            # left_child and right_child will be set later
        self.node_count += 1
        return 0
    cpdef  DOUBLE_t[:,:] local_variable_importance_g(self, object X, object y, object grouping):
        return self._parse(X, y, grouping)

    cdef inline DOUBLE_t[:,:] _parse(self, object X, np.ndarray y, np.ndarray grouping):
        cdef:
            DTYPE_t[:, :] X_ndarray = X
            DOUBLE_t[:] y_ndarray = y
            INT32_t[:,:] group = grouping
            DOUBLE_t[:,:] node_subs_y_sum
            DOUBLE_t[:,:] node_subs_squared_y_sum
            SIZE_t[:,:] node_subs_sample_num
            DOUBLE_t[:,:] node_impurity
            DOUBLE_t[:,:] importances
            SIZE_t subs_num = group.shape[0]
            SIZE_t sample_num = X_ndarray.shape[0]
            SIZE_t subs_i = 0
            SIZE_t sample_i
            SIZE_t node_i
            SIZE_t present_node_id
            ## refering to the struct
            SIZE_t node_num = self.node_count
            Node* node = NULL
            Node* left
            Node* right
        importances = np.zeros((subs_num, self.n_features))
        node_subs_y_sum = np.zeros((subs_num, node_num))
        node_subs_squared_y_sum = np.zeros((subs_num, node_num))
        node_subs_sample_num = np.zeros((subs_num, node_num), dtype = np.intp)
        node_impurity = np.zeros((subs_num, node_num))
        with nogil:
            for subs_i in range(subs_num):
                for sample_i in range(sample_num):
                    if group[subs_i, sample_i]:
                        present_node_id = 0
                        node = self.nodes
                        # While node not a leaf
                        while node.left_child != _TREE_LEAF:
                            # ... and node.right_child != _TREE_LEAF:
                            node_subs_y_sum[subs_i, present_node_id] += y_ndarray[sample_i]
                            node_subs_squared_y_sum[subs_i, present_node_id] += (y_ndarray[sample_i])**2
                            node_subs_sample_num[subs_i, present_node_id] += 1
                            if X_ndarray[sample_i, node.feature] <= node.threshold:
                                present_node_id = node.left_child
                                node = &self.nodes[present_node_id]
                            else:
                                present_node_id = node.right_child
                                node = &self.nodes[present_node_id]
            ## compute local impurty for first node
            node = self.nodes
            for node_i in range(node_num):
                for subs_i in range(subs_num):
                    if node_subs_sample_num[subs_i, node_i] > 1:
                        node_impurity[subs_i, node_i] = (node_subs_squared_y_sum[subs_i, node_i] -
                            node_subs_y_sum[subs_i, node_i]**2/node_subs_sample_num[subs_i, node_i])
                node += 1
            ## compute feature importance
            node = self.nodes
            for node_i in range(node_num):
                for subs_i in range(subs_num):
                    if node.left_child != _TREE_LEAF and node_subs_sample_num[subs_i, node_i] > 1:
                        importances[subs_i, node.feature] = (node_impurity[subs_i, node_i] -
                            node_impurity[subs_i, node.left_child]- node_impurity[subs_i, node.right_child])
        return (node_impurity)






