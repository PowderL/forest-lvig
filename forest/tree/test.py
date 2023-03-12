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
        SIZE_t subs_num = group.shape[0]a
        SIZE_t sample_num = X_ndarray.shape[0]
        SIZE_t subs_i = 0
        SIZE_t sample_i
        SIZE_t node_i
        SIZE_t present_node_id
        ## refering to the struct
        SIZE_t node_num = self.node_count
        Node* node = NULL
        #Node* left
        #Node* right
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
                    while True:
                        # ... and node.right_child != _TREE_LEAF:
                        node_subs_y_sum[subs_i, present_node_id] += y_ndarray[sample_i]
                        node_subs_squared_y_sum[subs_i, present_node_id] += (y_ndarray[sample_i])**2
                        node_subs_sample_num[subs_i, present_node_id] += 1
                        if node.left_child == _TREE_LEAF:
                            break
                        if X_ndarray[sample_i, node.feature] <= node.threshold:
                            present_node_id = node.left_child
                            node = &self.nodes[present_node_id]
                        else:
                            present_node_id = node.right_child
                            node = &self.nodes[present_node_id]
        ## compute local impurty for first node
        for node_i in range(node_num):
            for subs_i in range(subs_num):
                if node_subs_sample_num[subs_i, node_i] > 1:
                    node_impurity[subs_i, node_i] = (node_subs_squared_y_sum[subs_i, node_i] -
                        (node_subs_y_sum[subs_i, node_i])**2/node_subs_sample_num[subs_i, node_i])

        ## compute feature importance
        for node_i in range(node_num):
            node = &self.nodes[node_i]
            for subs_i in range(subs_num):
                if node.left_child != _TREE_LEAF and node_subs_sample_num[subs_i, node_i] > 1:
                    importances[subs_i, node.feature] += (node_impurity[subs_i, node_i] -
                        node_impurity[subs_i, node.left_child]- node_impurity[subs_i, node.right_child])

    return (importances)
