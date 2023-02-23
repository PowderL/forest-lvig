from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
import numpy as np
import pandas as pd


class ForestRegressor():
    def __init__(self, model, X_varnames, n_jobs=None, verbose=0):
        self.model = model
        self.max_depth = model.max_depth
        self.max_depth = 10 if model.max_depth is None else self.max_depth
        # self.X_varnames = X_varnames
        self.n_features = len(X_varnames)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def lvig_base(self, base_estimators_, X, y, partition_feature=None, method="lvig_based_impurity_cython_version",  norm=True):
        '''
        :param X:
        :param y:
        :param partition_feature:
        :param method: must one of ["lvig_based_impurity","lvig_based_accuracy",
        "lvig_based_impurty_cython_version"]
        :param norm:
        :return:
        '''
        columns = X.columns
        if method not in ["lvig_based_impurity", "lvig_based_accuracy",
                          "lvig_based_impurity_cython_version"]:
            raise ValueError('''method must one of [lvig_based_impurity, 
                lvig_based_accuracy, lvig_based_impurity_cython_version] 
                instead of %s''' % (method))
        X = np.asarray(X).astype("float32")
        y = np.asarray(y).ravel().astype("float64")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The input X and y have different number of instance")
        if X.shape[1] != self.n_features:
            raise ValueError(
                'The input X has different number of features with the trained model')
        if partition_feature is not None:
            partition_feature = np.asarray(partition_feature).ravel()
            if X.shape[0] != partition_feature.shape[0]:
                raise ValueError(
                    'The input X has different number of instances with partion_feature variable')
            unique_element = np.unique(partition_feature)
            subspace_flag = unique_element[:, None] == partition_feature
        else:
            unique_element = None
            subspace_flag = np.ones((X.shape[0]))
        if method == "lvig_based_impurity_cython_version":
            subspace_flag = subspace_flag.astype("int32")
        if not isinstance(norm, bool):
            raise ValueError(
                "Variable norm must True or False, but get %s" % s(norm))
        # Parallel loop
        if method == "lvig_based_impurity":
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               **_joblib_parallel_args(prefer="threads"))(
                delayed(tree.lvig_based_impurity)(
                    base_estimators_[i], X, y, subspace_flag)
                for i, tree in enumerate(self.estimators_))
        elif method == "lvig_based_accuracy":
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               **_joblib_parallel_args(prefer='threads'))(
                delayed(tree.lvig_based_accuracy)(X, y, subspace_flag)
                for tree in self.estimators_)  # traverse each tree in a forest
        else:
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               **_joblib_parallel_args(prefer='threads'))(
                delayed(tree.lvig_based_impurity_cython_version)(
                    X, y, subspace_flag)
                for tree in self.estimators_)  # traverse each tree in a forest
        # Vertically stack the arrays returned by traverse forming a
        feature_importances = np.vstack(results)
        # To compute weighted feature importance
        feature_importances_re = np.average(feature_importances, axis=0)
        if norm:  # whether standardise the output
            sum_of_importance_re = feature_importances_re.sum(
                axis=1).reshape(feature_importances_re.shape[0], 1)
            feature_importances_re = feature_importances_re / \
                (sum_of_importance_re+(sum_of_importance_re == 0))
        return pd.DataFrame(feature_importances_re, columns=columns, index=unique_element)
