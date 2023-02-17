import sys
import os

from forest._build_utils import cythonize_extensions


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy
    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('forest', parent_package, top_path)
    # submodules with build utilities
    # config.add_subpackage('__check_build')
    config.add_subpackage('_build_utils')
    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    # config.add_subpackage('ensemble/_hist_gradient_boosting')
    config.add_subpackage('ensemble')
    # config.add_subpackage('metrics')
    # config.add_subpackage('dependency')
    config.add_subpackage('tree')
    # config.add_subpackage('utils')
    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
