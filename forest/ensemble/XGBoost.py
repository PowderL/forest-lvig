
from ..tree import TreeRegressor
import json
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
import numpy as np
from ._forest import ForestRegressor


class LVIG_XGBoostRegressor(ForestRegressor):
    def __init__(self, model, X_varnames, n_jobs=None, verbose=0):
        super().__init__(model, X_varnames, n_jobs=n_jobs, verbose=verbose)
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
        return (self.lvig_base(X, y, partition_feature, method="lvig_based_impurity_cython_version",  norm=norm))
