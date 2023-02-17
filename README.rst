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

Compute local variable importance based on decrease in node impurity ::

	# use Boston house-price datasets as an example
	from sklearn.datasets import load_boston
	train_x, train_y = load_boston(return_X_y=True)
	# partition_feature could a column from train_x
	partition_feature = train_x[:, 1]
	from forest.ensemble import RandomForestRegressor
	rf = RandomForestRegressor(500, max_features=0.3)
	rf.fit(train_x, train_y)
	local_variable_importance = rf.lvig(train_x, train_y, partition_feature = partition_feature,
    	method = "lvig_based_impurity")
	
or compute local variable importance based on decrease in accuracy ::

	local_variable_importance = rf.lvig(train_x, train_y, partition_feature = partition_feature,
            method = "lvig_based_accuracy")

to achieve lower computation cost, we provide a cython_ version based on decrease in node impurity ::
    
	local_variable_importance = rf.lvig(train_x, train_y, partition_feature = partition_feature,
    	method = "lvig_based_impurity_cython_version")

