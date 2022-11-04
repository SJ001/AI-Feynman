from numpy.distutils.core import Extension, setup
from Cython.Build import cythonize

source_files = Extension(
                        name="cython_wrapper",
                        sources=["aifeynman/cython_wrapper.pyx", "aifeynman/bruteforce.cpp"],
                        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                        language='c++',
                        extra_compile_args=["-O3", "-Wall"] # add "-pg" for profiler support
                        )

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='aifeynman',
    version='2.1.0',
    description='AI Feynman: a Physics-Inspired Method for Symbolic Regression',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/SJ001/AI-Feynman',
    author='Silviu-Marian Udrescu',
    author_email='sudrescu@mit.edu',
    license='MIT',
    packages=['aifeynman'],
    package_dir={'aifeynman': 'aifeynman'},
    package_data={'aifeynman': ['*.txt', '*.cpp']},
    include_package_data=True,
    ext_modules=cythonize(source_files, force=True),
    python_requires='>3.6',
    install_requires=['matplotlib',
                      'numpy',
                      'sklearn',
                      'sortedcontainers',
                      'sympy == 1.4',
                      'torch >= 1.4.0',
                      'pandas',
                      'scipy',
                      'tqdm >= 4.60.0',
                      'cython'
                      ]
)
