from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
    name='box overlaps',
    ext_modules=cythonize('box_overlaps.pyx'),
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext}
)