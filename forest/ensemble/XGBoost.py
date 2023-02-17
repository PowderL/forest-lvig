
from ..tree import TreeRegressor
import json
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
import numpy as np


class LVIG_XGBoostRegressor:
    def __init__(self, model, X_varnames, n_jobs=None, verbose=0):
        self.model = model
        self.max_depth = model.max_depth
        self.X_varnames = X_varnames
        self.n_features = len(X_varnames)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.init_model()

    def init_model(self):
        self.num_trees = self.model.n_estimators
        tree_attrs_list = self.model._Booster.get_dump(dump_format="json")
        tree_attrs_list = [json.loads(tree_attr)
                           for tree_attr in tree_attrs_list]
        # 需要把这个
        # 需要提前做一些转换
        tree_attrs_list = [self.replace_var_names(
            tree_attrs) for tree_attrs in tree_attrs_list]
        self.estimators_ = [TreeRegressor(self.n_features, self.max_depth)
                            for i in range(self.num_trees)]

        Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(prefer='threads'))(
            delayed(tree.rebuild_tree_xgb)(tree_attrs_list[i])
            for i, tree in enumerate(self.estimators_))

    def replace_var_names(self, tree_attrs):
        for key, value in tree_attrs.items():
            if key == "split":
                tree_attrs['split'] = self.X_varnames.index(
                    tree_attrs['split'])
            if key == "children":
                tree_attrs["children"][0] = self.replace_var_names(
                    value[0])
                tree_attrs["children"][1] = self.replace_var_names(
                    value[1])
        return (tree_attrs)

    def lvig(self, X, y, partition_feature=None, norm=True):
        '''
        :param X:
        :param y:
        :param partition_feature:
        :param method: must one of ["lvig_based_impurity","lvig_based_accuracy",
        "lvig_based_impurty_cython_version"]
        :param norm:
        :return:
        '''
        import pandas as pd
        columns = [i for i in range(self.n_features)]
        method = "lvig_based_impurity_cython_version"
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
                delayed(tree.lvig_based_impurity)(X, y, subspace_flag)
                for tree in self.estimators_)
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
        # return pd.DataFrame(feature_importances_re, columns=columns, index=unique_element)
        return pd.DataFrame(feature_importances_re)
