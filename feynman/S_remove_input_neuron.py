# Remove on input neuron from a NN

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import pickle
from matplotlib import pyplot as plt
import torch.utils.data as utils
import time
import os

is_cuda = torch.cuda.is_available()

def remove_input_neuron(net,n_inp,idx_neuron,ct_median,save_filename):
    removed_weights = net.linear1.weight[:,idx_neuron]
    # Remove the weights associated with the removed input neuron                                                                                                               
    t = torch.transpose(net.linear1.weight,0,1)
    preserved_ids = torch.LongTensor(np.array(list(set(range(n_inp)) - set([idx_neuron]))))
    t = nn.Parameter(t[preserved_ids, :])
    net.linear1.weight = nn.Parameter(torch.transpose(t,0,1))
    # Adjust the biases
    if is_cuda:
        net.linear1.bias = nn.Parameter(net.linear1.bias+torch.tensor(ct_median*removed_weights).float().cuda())
    else:
        net.linear1.bias = nn.Parameter(net.linear1.bias+torch.tensor(ct_median*removed_weights).float())
    torch.save(net.state_dict(), save_filename)

