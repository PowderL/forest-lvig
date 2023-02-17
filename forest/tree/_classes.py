import numbers
import warnings
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

import numpy as np
from scipy.sparse import issparse, csr_matrix
from numpy.random import permutation

from ._tree import Tree


__all__ = ["TreeRegressor"]


class TreeRegressor:
    """Base class for decision trees.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(self, n_features, max_depth):
        self.n_features = n_features
        self.max_depth = max_depth
        self.tree_ = Tree(int(n_features), int(max_depth))

    def rebuild_tree_xgb(self, tree_attrs):
        leaf = tree_attrs.get("leaf", None)
        node_id = tree_attrs.get("nodeid")
        left_child = tree_attrs.get("yes")
        right_child = tree_attrs.get("no")
        threshold = tree_attrs.get("split_condition")
        feature = tree_attrs.get("split")
        is_leaf = False
        if leaf is not None:
            is_leaf = True
        if is_leaf:
            self.tree_._rebuild_node(node_id, is_leaf, -1, -1, -1.0, -1)
        else:
            self.tree_._rebuild_node(
                node_id, is_leaf, left_child, right_child, threshold, feature)
            self.rebuild_tree_xgb(tree_attrs['children'][0])
            self.rebuild_tree_xgb(tree_attrs['children'][1])

    def rebuild_tree_rfr(self, tree):
        '''
        reconstruct the random forest.
        '''
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        for node_id in range(children_left.shape[0]):
            left_child_i = children_left[node_id]
            right_child_i = children_right[node_id]
            feature_i = feature[node_id]
            threshold_i = threshold[node_id]
            is_leaf = False if left_child_i > 0 else True
            self.tree_._rebuild_node(node_id, is_leaf, left_child_i,
                                     right_child_i, threshold_i, feature_i)

    def lvig_based_impurity(self, X, y, subspace_flag):
        # this code is for the dubug of the cython version
        '''
        This function is to compute variable importance for each local group in one tree.
        It returns a 1*n_group*n_feature 3-d array containing the variable importance
        of all variables for each subspace
        :param X: X of input data
        :param y: Y of input data
        :param subspace_flag: a array contains bool value used for selecting rows
        :return:a 1*n_group*n_feature 3-d array containing the variable importance
        of all variables for each group in this tree
        '''
        # to obtain squared y
        y = csr_matrix(y)
        y_squared = y.multiply(y)
        feature_index = np.arange(0, X.shape[1])
        # to obtain the split feature at each node in this tree
        node_split_feature = self.tree_.feature
        # reshape it
        node_split_feature_copy = node_split_feature.reshape(
            node_split_feature.shape[0], 1)
        # use sparse matrix to contain to feature array
        fa_feature = csr_matrix(node_split_feature_copy == feature_index)
        # to obtain the index of left child node for each split node
        children_left = self.tree_.children_left
        # to obtain the index of right child node for each split node
        children_right = self.tree_.children_right
        # generate a zero matrix to contain the results
        TVI = np.zeros((1, subspace_flag.shape[0], X.shape[1]))
        for inde in range(subspace_flag.shape[0]):
            # to obtain the index of row
            subspace_index = np.where(subspace_flag[inde])[0]
            # via index of row to select rows of X
            choose_X = X[subspace_index]
            # via index of row to select rows of Y
            choose_y = y[0, subspace_index]
            # via index of row to select rows of squared Y
            choose_y_squared = y_squared[0, subspace_index]
            # to obtain decision path for each row
            decision_path_v = self.decision_path(choose_X)
            # to obtain the numbers of records at each node in this tree
            data_num = np.array(decision_path_v.sum(axis=0))[0]
            # to select node which contain records more than one record. Because there is no squared error if the node has less than tow records.
            data_num_lar = (data_num > 1)
            # to obtain the index of node which containing more one record.
            col_choose = np.where(data_num_lar)[0]
            # select nodes
            decision_path_pre = decision_path_v[:, col_choose]
            # compute sum of y for each selected nodes
            node_y_sum = choose_y.dot(decision_path_pre)
            # to compute squared sum of y for each selected nodes
            node_y_sum_squared = node_y_sum.multiply(node_y_sum)
            # compute sum of squared y for every selected nodes
            node_y_squared_sum = choose_y_squared.dot(decision_path_pre)
            # get sum of squared error (SSE) for each node.
            node_sse_v = node_y_squared_sum - \
                node_y_sum_squared / data_num[col_choose]
            node_sse_v = np.array(node_sse_v)[0]
            node_sse_parent = node_sse_v
            # select nodes index of left child node for nodes which have more than one records
            choose_children_left = children_left[col_choose]
            # select nodes index of right child node for nodes which have more than one records
            choose_children_right = children_right[col_choose]
            left_inter_sse = np.intersect1d(
                col_choose, choose_children_left, return_indices=True)
            right_inter_sse = np.intersect1d(
                col_choose, choose_children_right, return_indices=True)
            # get the index of 'col_choose' whose  element is in 'choose_children_left'
            searchsort_sse_left = left_inter_sse[1]
            # get the index of 'col_choose' whose  element is in 'choose_children_right'
            searchsort_sse_right = right_inter_sse[1]
            # get the index of 'choose_children_left' whose  element is in 'col_choose'
            node_left_index = left_inter_sse[2]
            # get the index of 'choose_children_right' whose element is in 'col_choose'
            node_right_index = right_inter_sse[2]
            # generate containers for left child nodes and right child nodes
            left_sse = np.zeros(node_sse_v.shape)
            right_sse = left_sse.copy()
            # assign SSE of left child nodes to the index where the corresponding parent node at
            left_sse[node_left_index] = node_sse_v[searchsort_sse_left]
            # assign SSE of right child nodes to the index where the corresponding parent node at
            right_sse[node_right_index] = node_sse_v[searchsort_sse_right]
            # parent node SSE minus left child node SSE and right child node SSE to get reduction of SSE (RSSE)
            node_rsse = node_sse_parent - left_sse - right_sse
            fa_feature_choose = fa_feature[col_choose]
            # sum up RSSE by split variable
            TVI[0, inde] = csr_matrix(node_rsse).dot(
                fa_feature_choose).toarray()
        return TVI

    def lvig_based_impurity_cython_version(self, X, y, subspace_flag):
        return (np.asarray(self.tree_.local_variable_importance_g(X, y, subspace_flag))[None, :, :])

    def lvig_based_accuracy(self, X, y, subspace_flag):
        '''
        This function is to compute local variable importance for each feature subspace data set in one tree.
        It returns a 1*n_group*n_feature 3-d array containing the variable importance of all variables for each subspace.
        :param X: X of input data
        :param y: Y of input data
        :param subpsace_flag: a array contains bool value used for selecting records if group_by
        is not None.
        :return:a 1*n_group*n_feature 3-d array containing the variable importance
        of all variables for each group
        '''

        # one-time getting the index for selecting records
        subspace_index = [np.where(subspace_flag_rows)[0]
                          for subspace_flag_rows in subspace_flag]
        # permutating the index for selecting records
        permutated_subspace_index = [permutation(
            subspace_index_rows) for subspace_index_rows in subspace_index]
        # compute sum of squared error (sse) before permutation for each record so-called original error
        sse = np.square(y-self.predict(X))
        # generate a array to contain the results
        tree_feature_importance_array = np.zeros(
            (1, subspace_flag.shape[0], X.shape[1]))
        # traverse all subspace data sets
        for each_set in range(subspace_flag.shape[0]):
            # select records without permutation
            un_permutated_x = X[subspace_index[each_set]]
            # select and permutate the records
            permutated_x_copy = X[permutated_subspace_index[each_set]].copy()
            # select original sse for the specified feature subspace dataset
            sse_sp_set = sse[subspace_index[each_set]]
            # select input Y
            subspace_y = y[subspace_index[each_set]]
            for feature_k in range(X.shape[1]):
                un_permutated_x_copy = un_permutated_x.copy()
                # get X whose variable at index "index2" is permutated
                un_permutated_x_copy[:,
                                     feature_k] = permutated_x_copy[:, feature_k]
                # get sum of squared error after permutation for the specified feature subspace data set and feature k
                spse_sp_set_k = np.square(
                    subspace_y-self.predict(un_permutated_x_copy))
                # the difference of mse before and after permutation is the variable importance for the
                TVI_set_k = np.mean(spse_sp_set_k - sse_sp_set)
                tree_feature_importance_array[0,
                                              each_set, feature_k] = TVI_set_k
        return tree_feature_importance_array

    def compute_feature_contribution(self, X, y):
        return(np.asarray(self.tree_.compute_feature_contribution(X, y))[None, :, :])

    def compute_feature_contribution_tree(self, X):
        return (np.asarray(self.tree_.compute_feature_contribution_tree(X))[None, :, :])
