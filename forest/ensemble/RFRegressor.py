from ..tree import TreeRegressor
import json
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
import numpy as np
from ._forest import ForestRegressor


class LVIG_RFRegressor(ForestRegressor):
    def __init__(self, model, X_varnames, n_jobs=None, verbose=0):
        super().__init__(model, X_varnames, n_jobs=n_jobs, verbose=verbose)
        self.init_model()

    def init_model(self):
        self.num_trees = self.model.n_estimators
        self.estimators_ = [TreeRegressor(self.n_features, self.max_depth)
                            for i in range(self.num_trees)]
        Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(prefer='threads'))(
            delayed(tree.rebuild_tree_rfr)(self.model.estimators_[i])
            for i, tree in enumerate(self.estimators_))

    def lvig(self, X, y, partition_feature=None, method="lvig_based_impurity_cython_version", norm=True):
        return (self.lvig_base(self.model.estimators_, X, y, partition_feature, method=method,  norm=norm))
