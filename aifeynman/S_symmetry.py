# checks for symmetries in the data

from __future__ import print_function
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
from .S_remove_input_neuron import remove_input_neuron
import time

is_cuda = torch.cuda.is_available()

class SimpleNet(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.linear1 = nn.Linear(ni, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64,64)
        self.linear5 = nn.Linear(64,1)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = self.linear5(x)
        return x

def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom

# checks if f(x,y)=f(x-y)
def check_translational_symmetry_minus(pathdir, filename):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+"/%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"/%s" %filename, usecols=(0,))

        if n_variables==1:
            print(filename, "just one variable for ADD \n")
            # if there is just one variable you have nothing to separate
            return (-1,-1,-1)
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+"/%s" %filename, usecols=(j,))
                variables = np.column_stack((variables,v))
        

        f_dependent = np.loadtxt(pathdir+"/%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():            
            # make the shift x->x+a for 2 variables at a time (different variables)
            min_error = 1000
            best_i = -1
            best_j = -1
            best_mu = 0
            best_sigma = 0
            for i in range(0,n_variables,1):
                for j in range(0,n_variables,1):
                    if i<j:
                        fact_translate = factors.clone()
                        a = 0.5*min(torch.std(fact_translate[:,i]),torch.std(fact_translate[:,j]))
                        fact_translate[:,i] = fact_translate[:,i] + a
                        fact_translate[:,j] = fact_translate[:,j] + a
                        list_errs = abs(product-model(fact_translate))
                        error = torch.median(list_errs)
                        mu = torch.mean(torch.log2(1+list_errs*2**30))
                        sigma = torch.std(torch.log2(1+list_errs*2**30))
                        if error<min_error:
                            min_error = error
                            best_i = i
                            best_j = j
                            best_mu = mu
                            best_sigma = sigma
        return min_error, best_i, best_j, best_mu, best_sigma

    except Exception as e:
        print(e)
        return (-1,-1,-1,-1,-1)
    
def do_translational_symmetry_minus(pathdir, filename, i,j):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+"/%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"/%s" %filename, usecols=(0,))

        for k in range(1,n_variables):
            v = np.loadtxt(pathdir+"/%s" %filename, usecols=(k,))
            variables = np.column_stack((variables,v))
        
        f_dependent = np.loadtxt(pathdir+"/%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():   
            file_name = filename + "-translated_minus"
            ct_median = torch.median(torch.from_numpy(variables[:,j]))
            data_translated = variables
            data_translated[:,i] = variables[:,i]-variables[:,j]
            data_translated =  np.delete(data_translated, j, axis=1)
            data_translated = np.column_stack((data_translated,f_dependent))
            try:
                os.mkdir("results/translated_data_minus/")
            except:
                pass
            np.savetxt("results/translated_data_minus/"+file_name , data_translated)
            remove_input_neuron(model,n_variables,j,ct_median,"results/NN_trained_models/models/"+filename + "-translated_minus_pretrained.h5")
            return ("results/translated_data_minus/",file_name)

    except Exception as e:
        print(e)
        return (-1,-1)
        

# checks if f(x,y)=f(x/y)
def check_translational_symmetry_divide(pathdir, filename):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+"/%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"/%s" %filename, usecols=(0,))

        if n_variables==1:
            print(filename, "just one variable for ADD \n")
            # if there is just one variable you have nothing to separate
            return (-1,-1,-1)
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+"/%s" %filename, usecols=(j,))
                variables = np.column_stack((variables,v))
        

        f_dependent = np.loadtxt(pathdir+"/%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():
            a = 1.2
            min_error = 1000
            best_i = -1
            best_j = -1
            best_mu = 0
            best_sigma = 0
            # make the shift x->x*a and y->y*a for 2 variables at a time (different variables)
            for i in range(0,n_variables,1):
                for j in range(0,n_variables,1):
                    if i<j:
                        fact_translate = factors.clone()
                        fact_translate[:,i] = fact_translate[:,i]*a
                        fact_translate[:,j] = fact_translate[:,j]*a
                        list_errs = abs(product-model(fact_translate))
                        error = torch.median(list_errs)
                        mu = torch.mean(torch.log2(1+list_errs*2**30))
                        sigma = torch.std(torch.log2(1+list_errs*2**30))
                        if error<min_error:
                            min_error = error
                            best_i = i
                            best_j = j
                            best_mu = mu
                            best_sigma = sigma
        return min_error, best_i, best_j, best_mu, best_sigma

    except Exception as e:
        print(e)
        return (-1,-1,-1,-1,-1)
    
    
def do_translational_symmetry_divide(pathdir, filename, i,j):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+"/%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"/%s" %filename, usecols=(0,))

        for k in range(1,n_variables):
            v = np.loadtxt(pathdir+"/%s" %filename, usecols=(k,))
            variables = np.column_stack((variables,v))
        
        f_dependent = np.loadtxt(pathdir+"/%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():             
            file_name = filename + "-translated_divide"
            data_translated = variables
            ct_median =torch.median(torch.from_numpy(variables[:,j]))
            data_translated[:,i] = variables[:,i]/variables[:,j]
            data_translated =  np.delete(data_translated, j, axis=1)
            data_translated = np.column_stack((data_translated,f_dependent))
            try:
                os.mkdir("results/translated_data_divide/")
            except:
                pass
            np.savetxt("results/translated_data_divide/"+file_name , data_translated)
            remove_input_neuron(model,n_variables,j,ct_median,"results/NN_trained_models/models/"+filename + "-translated_divide_pretrained.h5")
            return ("results/translated_data_divide/",file_name)

    except Exception as e:
        print(e)
        return (-1,1)

# checks if f(x,y)=f(x*y)
def check_translational_symmetry_multiply(pathdir, filename):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+"/%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"/%s" %filename, usecols=(0,))

        if n_variables==1:
            print(filename, "just one variable for ADD \n")
            # if there is just one variable you have nothing to separate
            return (-1,-1,-1)
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+"/%s" %filename, usecols=(j,))
                variables = np.column_stack((variables,v))
        

        f_dependent = np.loadtxt(pathdir+"/%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():
            a = 1.2
            min_error = 1000
            best_i = -1
            best_j = -1
            best_mu = 0
            best_sigma = 0
            # make the shift x->x*a and y->y/a for 2 variables at a time (different variables)
            for i in range(0,n_variables,1):
                for j in range(0,n_variables,1):
                    if i<j:
                        fact_translate = factors.clone()
                        fact_translate[:,i] = fact_translate[:,i]*a
                        fact_translate[:,j] = fact_translate[:,j]/a
                        list_errs = abs(product-model(fact_translate))
                        error = torch.median(list_errs)
                        mu = torch.mean(torch.log2(1+list_errs*2**30))
                        sigma = torch.std(torch.log2(1+list_errs*2**30))
                        if error<min_error:
                            min_error = error
                            best_i = i
                            best_j = j
                            best_mu = mu
                            best_sigma = sigma
        return min_error, best_i, best_j, best_mu, best_sigma

    except Exception as e:
        print(e)
        return (-1,-1,-1,-1,-1)

def do_translational_symmetry_multiply(pathdir, filename, i,j):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+"/%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"/%s" %filename, usecols=(0,))

        for k in range(1,n_variables):
            v = np.loadtxt(pathdir+"/%s" %filename, usecols=(k,))
            variables = np.column_stack((variables,v))
        
        f_dependent = np.loadtxt(pathdir+"/%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():             
            file_name = filename + "-translated_multiply"
            data_translated = variables
            ct_median =torch.median(torch.from_numpy(variables[:,j]))
            data_translated[:,i] = variables[:,i]*variables[:,j]
            data_translated =  np.delete(data_translated, j, axis=1)
            data_translated = np.column_stack((data_translated,f_dependent))
            try:
                os.mkdir("results/translated_data_multiply/")
            except:
                pass
            np.savetxt("results/translated_data_multiply/"+file_name , data_translated)
            remove_input_neuron(model,n_variables,j,ct_median,"results/NN_trained_models/models/"+filename + "-translated_multiply_pretrained.h5")
            return ("results/translated_data_multiply/",file_name)

    except Exception as e:
        print(e)
        return (-1,1)

# checks if f(x,y)=f(x+y)
def check_translational_symmetry_plus(pathdir, filename):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+"/%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"/%s" %filename, usecols=(0,))

        if n_variables==1:
            print(filename, "just one variable for ADD \n")
            # if there is just one variable you have nothing to separate
            return (-1,-1,-1)
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+"/%s" %filename, usecols=(j,))
                variables = np.column_stack((variables,v))
        

        f_dependent = np.loadtxt(pathdir+"/%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
                product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():
            min_error = 1000
            best_i = -1
            best_j = -1
            best_mu = 0
            best_sigma = 0
            for i in range(0,n_variables,1):
                for j in range(0,n_variables,1):
                    if i<j:
                        fact_translate = factors.clone()
                        a = 0.5*min(torch.std(fact_translate[:,i]),torch.std(fact_translate[:,j]))
                        fact_translate[:,i] = fact_translate[:,i] + a
                        fact_translate[:,j] = fact_translate[:,j] - a
                        list_errs = abs(product-model(fact_translate))
                        error = torch.median(list_errs)
                        mu = torch.mean(torch.log2(1+list_errs*2**30))
                        sigma = torch.std(torch.log2(1+list_errs*2**30))
                        if error<min_error:
                            min_error = error
                            best_i = i
                            best_j = j
                            best_mu = mu
                            best_sigma = sigma
        return min_error, best_i, best_j, best_mu, best_sigma

    except Exception as e:
        print(e)
        return (-1,-1,-1,-1,-1)
    
def do_translational_symmetry_plus(pathdir, filename, i,j):
    try:
        pathdir_weights = "results/NN_trained_models/models/"

        # load the data
        n_variables = np.loadtxt(pathdir+"/%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"/%s" %filename, usecols=(0,))

        for k in range(1,n_variables):
            v = np.loadtxt(pathdir+"/%s" %filename, usecols=(k,))
            variables = np.column_stack((variables,v))
        
        f_dependent = np.loadtxt(pathdir+"/%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        # load the trained model and put it in evaluation mode
        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
        model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        model.eval()

        models_one = []
        models_rest = []

        with torch.no_grad():             
            file_name = filename + "-translated_plus"
            data_translated = variables
            ct_median =torch.median(torch.from_numpy(variables[:,j]))
            data_translated[:,i] = variables[:,i]+variables[:,j]
            data_translated =  np.delete(data_translated, j, axis=1)
            data_translated = np.column_stack((data_translated,f_dependent))
            try:
                os.mkdir("results/translated_data_plus/")
            except:
                pass
            np.savetxt("results/translated_data_plus/"+file_name , data_translated)
            remove_input_neuron(model,n_variables,j,ct_median,"results/NN_trained_models/models/"+filename + "-translated_plus_pretrained.h5")
            return ("results/translated_data_plus/", file_name)
        
    except Exception as e:
        print(e)
        return (-1,-1)
