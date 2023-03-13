
'''
Algorithm based on  decrease in accuracy
'''
import numpy as np
import pandas as pd
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from numpy import float64
from numpy.random import permutation
from numpy import vstack, mean, square, array, zeros


class lvig:
    '''
        This class completely inherit scikit-learn's RandomForestRegressor.
        I provide two additional function to compute local variable importance
         importance. One is compute_feature_importance and another is traverse
         which helps compute_feature_importance to traverse every tree in the forest.
    '''

    def __init__(self, model):
        self.model = model

    def compute_feature_importance(self, x, y, partition_feature=None, norm=True):
        '''
        :param x: input X of data and must be Pandas.core.frame.DataFrame or Pandas.core.series.Series
        :param y: input Y of data and do not need specify the type, but must be supported in numpy
        :param partition_feature: used for partitioning the data into local data subspaces and must
        be a column of data that can be hashed, but is optional. You can partition the data in advance
        instead and input feature subspace one by one.
        For example, if you want to compute local variable importance for each
        day, you only need to let partition_feature = day of year (1-365).
        Or input the feature subspace for each day one by one.
        :param norm: Yes or No normalise the output leading to the sum of each row equals to one..
        :param n_jobs: The number of jobs paralleling at the same time. Please refer to class Parallel
        in package sklearn for more detailed information.
        :return: local variable importance
        '''
        # to obtain the names of variables
        if not isinstance(x, Series) and not isinstance(x, DataFrame):
            raise TypeError(
                "{0} must be pandas.core.frame.DataFrame or pandas.core.series.Series not {1}".format(x, type(x)))
        columns = x.columns
        # convert input X into numpy.array
        x = array(x, dtype=float64)
        # convert input Y to 1-D array
        y = array(y).ravel()
        # to obtain the number of variables
        self.FN = x.shape[1]
        # Produce data_choose array.This array contains bool values to choose rows for each feature subspace dataset
        if type(partition_feature) != type(None):
            partition_factor = list(partition_feature)
            # use set structure to extract factors
            partition_factor_set = set(partition_factor)
            partition_factor_list = list(partition_factor_set)
            # to obtain the number of group attribute
            self.FL = len(partition_factor_list)
            partition_factor_arr = np.array(
                partition_factor_list).reshape(self.FL, 1)
            # for each factor find out the rows of input group_by which is equal to it
            data_choose_bool = partition_factor_arr == partition_factor
        else:
            # if there is no group_by inputted, using all input rows
            self.FL = 1
            partition_factor_list = None
            data_choose_bool = np.ones((1, x.shape[0])) == 1
        # use the parallel in model rather algorithm parallel
        # one-time getting the index for selecting records
        data_choose_index = [np.where(data_choose_bool_rows)[
            0] for data_choose_bool_rows in data_choose_bool]
        sorted_index = np.hstack(
            [np.repeat(i, data_choose_index[i].shape[0]) for i in range(self.FL)])
        data_choose_index_combo = np.hstack(data_choose_index)
        original_x = x.copy()
        x = x[data_choose_index_combo, :]
        y = y[data_choose_index_combo]
        # permutating the index for selecting records
        permutated_choose_index = [permutation(
            data_choose_index_rows) for data_choose_index_rows in data_choose_index]
        permutated_choose_index_combo = np.hstack(permutated_choose_index)
        sse = square(y-self.model.predict(x))
        # generate a array to contain the results
        sse_increase_list = []
        for feature_k in range(self.FN):
            x_copy = x.copy()
            if feature_k == 11:
                #                 print(np.unique(x[sorted_index == 3, feature_k]))
                pass
            x_copy[:, feature_k] = original_x[permutated_choose_index_combo, feature_k]
            permutated_sse_k = square(y-self.model.predict(x_copy))
            # the difference of mse before and after permutation is the variable importance for the
            sse_increase_k = permutated_sse_k - sse
            sse_increase_list.append(sse_increase_k[:, None])
        sse_increase = np.hstack(sse_increase_list)
        sse_increase_df = pd.DataFrame(
            sse_increase, columns=columns, index=sorted_index)
        accuracy_increase_df = sse_increase_df.groupby(sorted_index).mean()
        accuracy_increase_df.index = partition_factor_list
        if not isinstance(norm, bool):
            raise TypeError(
                '{0} must be True or False not {1}'.format(norm, type(norm)))
        if norm:  # whether standardise the output
            # sum up each row
            sum_accuracy_increase_df = accuracy_increase_df.sum(axis=1)
            sum_accuracy_increase_df = np.array(
                sum_accuracy_increase_df)[:, None]
            # each one is divided by the sum of this row
            accuracy_increase_norm_df = accuracy_increase_df / \
                (sum_accuracy_increase_df+(sum_accuracy_increase_df == 0))
        else:
            # directly output without normalization
            accuracy_increase_norm_df = accuracy_increase_df
        # return the result with the form of DataFrame
        return accuracy_increase_norm_df
