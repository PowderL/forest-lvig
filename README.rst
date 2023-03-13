forest-gis
##########

|PythonVersion|_ |pypi|_ |Downloads|_

.. |Downloads| image:: https://pepy.tech/badge/forest-gis/month
.. _Downloads: https://pepy.tech/project/forest-gis/month
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue
.. |pypi| image:: https://badge.fury.io/py/forest-gis.svg
.. _pypi : https://pypi.org/project/forest-gis

Installation
^^^^^^^^^^^^

Dependencies
------------

forest-gis requires:

- Python (>= 3.6)
- NumPy (>= 1.15.0)
- SciPy (>= 0.19.1)
- joblib (>= 0.14)
- scikit-learn (>=0.19.0)

For Windwos
------------

If you already have a working installation of numpy and scipy,
and you plateform is Windows 32-bit or 64-bit, the easiest way 
to install forest-gis is using ``pip`` ::

    pip install -U forest-gis

or ``conda`` ::

    conda install -c conda-forge forest-gis

For linux
------------
At present, on the pypi_, we only provide wheel_ files supporting
Python3.6, 3.7, 3.8 for Windows 32-bit, Windows 64-bit. Though the
wheel_ files for Linux 64-bit are also provided, you may encouter
problems if your Linux system has a lower version of ``glibc`` than
ubantu 18.x because the wheel_ files was just compiled on ubantu 18.x
If you get wrong when use ``pip`` to install ``forest-gis``, you can
try to install "forest-gis" from source.

For macOS
------------
At present, install ``forest-gis``  from wheel_ files are not provied for macOS.

.. _wheel: https://wheel.readthedocs.io/en/stable
.. _pypi: https://pypi.org/project/forest-gis

Build forest-gis from source
----------------------------

For Windows and Linux

**Necessarily**, before you install the ``forest-gis`` from source, 
you need to first install or update cython_ and numpy_  to the newest
version and then run ::

    pip install cython
    pip install numpy
    pip install --verbose .

For macOS, first install the macOS command line tools ::
    
    brew install libomp
    
Set the following environment variables ::
    
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"

Finally, build forest-gis ::
    
    pip install --verbose .

.. _cython: https://cython.org/
.. _numpy: https://numpy.org/

User Guide
^^^^^^^^^^^^

Compute local variable importance based on the impurity metric ::

        # use Boston house-price datasets as an example
        from sklearn.datasets import load_boston
        train_x, train_y = load_boston(return_X_y=True)
        # partition_feature could a column from train_x
        partition_feature = train_x[:, 1]
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        from forest.ensemble import impurity_LVIG_RFRegressor
        from forest.ensemble import impurity_LVIG_EXTRegressor
        rf = RandomForestRegressor(500, max_features=0.3)
        rf.fit(train_x, train_y)
        ## using random forest model to compute local variable importance
        var_names = ["var_" + str(i) for i in range(train_x.shape[1])]
        lvig_handler = impurity_LVIG_RFRegressor(rf, var_names)
        local_variable_importance = lvig_handler.lvig(train_x, train_y, partition_feature = partition_feature)

        # use extra-trees to compute local variable importance
        model = ExtraTreesRegressor(500, max_features=0.3)
        model.fit(train_x, train_y)
        lvig_handler = impurity_LVIG_EXTRegressor(rf, var_names)
        local_variable_importance = lvig_handler.lvig(train_x, train_y, partition_feature = partition_feature)

or compute local variable importance based on the accuracy metric ::

        from forest.ensemble import accuracy_LVIG
        model = RandomForestRegressor(500, max_features=0.3)
        model.fit(train_x, train_y)
        lvig_handler = accuracy_LVIG(model)
        ## compute local variable importance
        ## 
        local_variable_importance = lvig_handler.compute_feature_importance(train_x, train_y, partition_feature = partition_feature)
        ## as the accuracy-based LVIG is a model-agnostic method, using other model like xgboost and gradient booting decission tree is applicable
        from sklearn.ensemble import GradientBoostingRegressor
        import xgboost as xgb
        ## based on gradient boosting decission tree
        model = GradientBoostingRegressor(n_estimators = 500, max_depth = 15, learning_rate=0.05, subsample=0.5, max_features=5)
        model.fit(train_x, train_y)
        lvig_handler = lvig(model)
        data = lvig_handler.compute_feature_importance(train_x, train_y, partition_feature)  

        ## based on xgboost
        model = xgb.XGBRegressor(n_estimators = 500, max_depth = 15, subsample = 0.5, eval_metric = "rmse", objective = "reg:linear", n_jobs=20, eta = 0.05, colsample_bynode = 0.33334)
        model.fit(train_x, train_y)
        lvig_handler = lvig(model)
        data = lvig_handler.compute_feature_importance(train_x, train_y, partition_feature)  

