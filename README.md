# Quick Start
## Installation
It's strongly recommended to setup a fresh virtual environment by typing

    virtualenv -p python3 feyn
    source feyn/bin/activate
    
First install numpy with `pip install numpy`.
The 'aifeynman' package is available on PyPI and can be installed with `pip install aifeynman`.

Note that for now, AI Feynman is supported only for Linux and Mac environments.

## First example
Move into a clean directory and run the following Python commands:

    import aifeynman

    aifeynman.get_demos("example_data") # Download examples from server
    aifeynman.run_aifeynman("./example_data/", "example1.txt", 60, "14ops.txt", polyfit_deg=3, NN_epochs=500)
    
This example will get solved in about 10-30 minutes depending on what computer you have and whether you have a GPU.

Here ‘example.txt’ contains the data table to perform symbolic regression on, with columns separated by spaces, commas or tabs. The other parameters control the search: here the brute-force modules tries combinations of the 14 basic operations in ‘14ops.txt’ for up to 60 seconds, polynomial fits are tried up to degree 3, and the interpolating neural network is trained for up to 500 epochs.

# AI-Feynman

This code is an improved implementation of AI Feynman: a Physics-Inspired Method for Symbolic Regression, Silviu-Marian Udrescu and Max Tegmark (2019) [[Science Advances](https://advances.sciencemag.org/content/6/16/eaay2631/tab-pdf)] and AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity, Udrescu S.M. et al. (2020) [[arXiv](https://arxiv.org/abs/2006.10782)].

Please check [this Medium article](https://towardsdatascience.com/ai-feynman-2-0-learning-regression-equations-from-data-3232151bd929) for a more detailed eplanation of how to get the code running.

In order to get started, run compile.sh to compile the fortran files used for the brute force code.

ai_feynman_example.py contains an example of running the code on some examples (found in the example_data directory). The examples correspond to the equations I.8.14, I.10.7 and I.50.26 in Table 4 in the paper. More data files on which the code can be tested on can be found in the [Feynman Symbolic Regression Database](https://space.mit.edu/home/tegmark/aifeynman.html). 

The main function of the code, called by the user, has the following parameters:

* pathdir - path to the directory containing the data file
* filename - the name of the file containing the data
* BF_try_time - time limit for each brute force call (set by default to 60 seconds)
* BF_ops_file_type - file containing the symbols to be used in the brute force code (set by default to "14ops.txt")
* polyfit_deg - maximum degree of the polynomial tried by the polynomial fit routine (set be default to 4)
* NN_epochs - number of epochs for the training (set by default to 4000)
* vars_name - name of the variables appearing in the equation (inluding the name ofthe output variable). This should be passed as a list of strings, with the name of the variables appearing in the same order as they are in the file containing the data
* test_percentage - percentage of the input data to be kept aside and used as the test set

The data file to be analyzed should be a text file with each column containing the numerical values of each (dependent and independent) variable. The solution file will be saved in the directory called "results" under the name solution_{filename}. The solution file will contain several rows (corresponding to each point on the Pareto frontier), each row showing: 

* the mean logarithm in based 2 of the error of the discovered equation applied to the input data (this can be though of as the average error in bits)
* the cummulative logarithm in based 2 of the error of the discovered equation applied to the input data (this can be though of as the cummulative error in bits)
* the complexity of the discovered equation (in bits)
* the error of the discovered equation applied to the input data
* the symbolic expression of the discovered equation

If test_percentage is different than zero, one more number is added in the beginning of each row, showing the error of the discovered equation on the test set.

ai_feynman_terminal_example.py allows calling the aiFeynman function from the command line. 
(e.g. python ai_feynman_terminal_example.py --pathdir=../example_data/ --filename=example1.txt). Use python ai_feynman_terminal_example.py --help to display all the available parameters that can be passed to the function.

# Citation

If you compare with, build on, or use aspects of the AI Feynman work, please cite the following:

```
@article{udrescu2020ai,
  title={AI Feynman: A physics-inspired method for symbolic regression},
  author={Udrescu, Silviu-Marian and Tegmark, Max},
  journal={Science Advances},
  volume={6},
  number={16},
  pages={eaay2631},
  year={2020},
  publisher={American Association for the Advancement of Science}
}
```

```
@article{udrescu2020ai,
  title={AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity},
  author={Udrescu, Silviu-Marian and Tan, Andrew and Feng, Jiahai and Neto, Orisvaldo and Wu, Tailin and Tegmark, Max},
  journal={arXiv preprint arXiv:2006.10782},
  year={2020}
}
```
