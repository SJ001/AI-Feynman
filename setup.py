import setuptools
from numpy.distutils.core import Extension, setup

sr1 = Extension(name='feynman._symbolic_regress1', sources=[
    'feynman/symbolic_regress1.f90'])

sr2 = Extension(name='feynman._symbolic_regress2', sources=[
    'feynman/symbolic_regress2.f90'])

sr3 = Extension(name='feynman._symbolic_regress3', sources=[
    'feynman/symbolic_regress3.f90'])


setup(
    name='feynman',
    version='2.0.0',
    description='AI Feynman: a Physics-Inspired Method for Symbolic Regression',
    url='https://github.com/SJ001/AI-Feynman',
    author='Silviu-Marian Udrescu',
    author_email='sudrescu@mit.edu',
    license='MIT',
    packages=['feynman'],
    package_dir={'feynman': 'feynman'},
    package_data={'feynman': ['*.txt', '*.f90']},
    include_package_data=True,
    ext_modules=[sr1, sr2, sr3],
    python_requires='>3.6',
    install_requires=['matplotlib',
                      'numpy',
                      'seaborn',
                      'sklearn',
                      'sortedcontainers',
                      'sympy >= 1.4',
                      'torch >= 1.4.0',
                      'torchvision',
                      ],
    entry_points={
        'console_scripts': [
            'feynman_sr1 = feynman._symbolic_regress1:go',
            'feynman_sr2 = feynman._symbolic_regress2:go',
            'feynman_sr3 = feynman._symbolic_regress3:go',
        ]}
)
