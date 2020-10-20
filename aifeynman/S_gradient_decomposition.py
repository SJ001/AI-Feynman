import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys
import torch
import torch.nn as nn

from itertools import chain, combinations, islice
from scipy.stats import mannwhitneyu
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr


from collections import Counter, namedtuple, OrderedDict
from copy import deepcopy

import torch.utils.data as data_utils


'''
This part discovers symmetry using the model fitted
on data
'''

def powerset_atleast_2(iterable, max_subset_size):
    "powerset([1,2,3]) -->  (1,2) (1,3) (2,3)"
    s = list(iterable)
    r = chain.from_iterable(combinations(s, r) for r in range(2, max_subset_size+1))
    # return islice(r, 0, 3)
    return r

def evaluate_derivatives(model, s,  pts):
    
    pts = pts.clone().detach()
    try:
        device = 'cuda' if model.is_cuda else 'cpu'
    except:
        device = 'cpu'
    pts = pts.to(device=device)

    pts.requires_grad_(True)
    outs = torch.zeros(pts.shape, device=device)
    if pts.grad is not None:
        pts.grad.data.zero_()
    for i in range(pts.shape[0]):
        model.eval()
        d = pts[[i], :]
        d.requires_grad_(True)
        r = model(d)
        r.backward()
    return pts.grad[:, s].clone().detach()

def evaluate_derivatives_andrew(model, s, pts):
    pts = pts.clone().detach()
    is_cuda = torch.cuda.is_available()
    grad_weights = torch.ones(pts.shape[0], 1)
    if is_cuda:
        pts = pts.cuda()
        model = model.cuda()
        grad_weights = grad_weights.cuda()
    
    pts.requires_grad_(True)
    outs = model(pts)
    grad = torch.autograd.grad(outs, pts, grad_outputs=grad_weights, create_graph=True)[0]
    return grad[:, s].detach()

def build_true_model(func):
    class TrueModel(nn.Module):
        def forward(self, X):
            return func(*[X[:,[i]] for i in range(X.shape[1])])
    return TrueModel()


def draw_samples(X, y, model, s, NUM_SAMPLES, point = None):
    '''
    Draw samples by sampling each dimension independently, 
    keeping the positions at s fixed to given point if exists,
    sampled point if not.
    '''
    n = X.shape[1]
    is_cuda = X.is_cuda
    device = 'cuda' if is_cuda else 'cpu'
    if point is None:
        idx = torch.randint(0, X.shape[0], (1,)).repeat(len(s))
    else:
        idx = torch.tensor([point]).repeat(len(s))
    pts = torch.randint(0, X.shape[0], (NUM_SAMPLES, X.shape[1]))
    pts[:, s] = idx
    actual_pts = torch.zeros(NUM_SAMPLES, X.shape[1], dtype=torch.float)
    for i in range(NUM_SAMPLES):
        for j in range(X.shape[1]):
            actual_pts[i,j] = X[pts[i,j],j]
    return actual_pts.to(device=device)

def visualize_score(normalized_grads, overall_score):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    ax.scatter([t[0] for t in normalized_grads], [t[1] for t in normalized_grads])
    ax.set_title(f'Score: {1-overall_score}')
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    plt.show()

def score_consistency(grads_tensor):
    n_pts = grads_tensor.shape[0]
    grads_tensor = grads_tensor.cpu()
    norms = [np.linalg.norm(grads_tensor[i, :].numpy()) for i in range(n_pts)]
    normalized_grads = [grads_tensor[i,:].numpy()/norms[i] for i in range(n_pts)]
    A = np.array(normalized_grads)
    D = np.einsum('ij,ik', A, A)
    evals, evecs = np.linalg.eig(D)
    nv = evals.shape[0]
    assert(nv == A.shape[1])
    # evecs should be d x (num eigenvectors - probably also d)
    dots = np.einsum('ij,jk->ik', A, evecs)
    scores = np.einsum('ij,ij->j', dots, dots)
    overall_score = np.max(scores)/n_pts
    mean_vec = evecs[:,np.argmax(scores)].flatten()
    return overall_score, mean_vec


plot_counter = 0
def visualize_distribution(hypot, bench):
    global plot_counter
    eps=1e-10
    hypot=np.array(hypot)
    bench = np.array(bench)
    kde = KernelDensity(kernel='tophat', bandwidth=0.25).fit(np.log10(1-hypot+eps)[:,np.newaxis])
    kde2 = KernelDensity(kernel='tophat', bandwidth=0.25).fit(np.log10(1-bench+eps)[:,np.newaxis])
    X_plot = np.linspace(-8, 0, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    log_dens2= kde2.score_samples(X_plot)
    plt.plot(X_plot[:, 0], np.exp(log_dens), color='#FFAA00', label='Hypothesis') #hypot is orange
    plt.plot(X_plot[:, 0], np.exp(log_dens2), color='#00AAFF', label='Benchmark')#blue is bench
    plt.legend()
    plt.savefig(f'plot_{plot_counter}.pdf')
    plot_counter += 1
    plt.show()

def get_kde(X_vals, dist, kernel='tophat', bandwidth=0.3):
    eps=1e-15
    scores = np.array(dist)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(np.log10(1-scores+eps)[:,np.newaxis])
    log_dens = kde.score_samples(X_vals)
    return np.exp(log_dens)

def signal_to_noise(high, low):
    '''
    Parameters: list of scores for hypothesis, list of scores for benchmark
    Returns a non-negative value
    '''
    eps = 1e-10
    def transform_scores(scores):
        scores = np.array(scores)
        scores = np.log10(1-scores + eps)
        return scores
    high = transform_scores(high)
    low = transform_scores(low)
    low_tail  = iqr(low, rng=(10,50))
    low_med  = np.median(low)
    high_med  = np.median(high)
    # print(high_med, low_med, low_tail)

    if high_med >= low_med:
        return 0
    else:
        # return (low_med - high_med) / low_tail
        return 0 - high_med

score_distributions = {}

def filter_decompositions_relative_scoring(X, y, model, max_subset_size=None, visualize=False):
    '''
    X: torch tensor. N * d
    y: torch tensor. N * 1
    model: torch nn.module
    returns: [(score, subset)],
    where subset is a tuple of indices in ascending order
    '''
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        X = X.cuda()
        y = y.cuda()
        model = model.cuda()
        # print('Using cuda')

    n = X.shape[1]
    NUM_TRIALS = 200
    NUM_SAMPLES =  60
    results = []
    max_subset_size = n-1 if max_subset_size is None else max_subset_size
    random_indices = np.random.choice(X.shape[0], size=min(NUM_TRIALS, X.shape[0]), replace=False)
    for s in powerset_atleast_2(range(0, n), max_subset_size=max_subset_size):
        print(f'Trying {s}')
        inv_s = tuple([i for i in range(0, n) if i not in s])
        hypot_scores = []
        bench_scores = []
        for i in range(random_indices.shape[0]):
            samples = draw_samples(X, y, model, s, NUM_SAMPLES, point=random_indices[i])
            score, _ = score_consistency(evaluate_derivatives_andrew(model, s, samples))
            hypot_scores.append(score)
        for i in range(random_indices.shape[0]):
            samples = draw_samples(X, y, model, (), NUM_SAMPLES, point=random_indices[i])
            score, _ = score_consistency(evaluate_derivatives_andrew(model, s, samples))
            bench_scores.append(score)
        snr = signal_to_noise(hypot_scores, bench_scores)
        # penalizes larger decompositions
        snr -= np.log10(2)*len(s) 
        results.append((snr, s))
        print((snr, s))
        if visualize:
            print(snr)
            visualize_distribution(hypot_scores, bench_scores)
            score_distributions[s] = (snr, hypot_scores)
    results.sort(key=lambda x:-x[0])
    return results

'''
Some helper functions
'''

def invert_subset(s, n):
    return tuple([i for i in range(n) if i not in s])
def to_bin_str(s, n):
    return ''.join(['1' if i in s else '0' for i in range(n)])
def to_numpy_mask(s, n):
    return np.array([i in s for i in range(n)])


def extract_gradients(X, y, model, s, num_points):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        X = X.cuda()
        y = y.cuda()
        model = model.cuda()
    idx = np.random.randint(0, X.shape[0], size=num_points)
    gradients = np.zeros((num_points, len(s)))
    for i in range(num_points):
        samples = draw_samples(X, y, model, s, 50, point=idx[i])
        _, gradients[i, :] = score_consistency(evaluate_derivatives_andrew(model, s, samples))
        # normalize first dimension
        if(gradients[i, 0] < 0):
             gradients[i,:] *= -1
    return np.concatenate((X[idx, :][:, s].cpu().data.numpy(), gradients),axis=1)


'''
Actual hook
Returns pair (numpy with 2k columns. First k are data points, next k are gradients), (bitmask as numpy array)
'''

def identify_decompositions(pathdir,filename, model, max_subset_size=2, visualize=False):
    print("identify_decompositions",pathdir,filename)
    data = np.loadtxt(pathdir+filename)
    X = torch.Tensor(data[:, :-1])
    y = torch.Tensor(data[:, [-1]])
    # Return best decomposition                                                                                                                                                   
    all_scores = filter_decompositions_relative_scoring(X, y, model, visualize=visualize)
    assert(all_scores)
    best_decomposition = all_scores[0][1]
    gradients = extract_gradients(X, y, model, best_decomposition, 10000)
    np.savetxt("results/gradients_gen_sym_%s" %filename, gradients)
    ll = np.arange(0,X.shape[1],1)
    print("mask", to_numpy_mask(best_decomposition, X.shape[1]))
    return ll[to_numpy_mask(best_decomposition, X.shape[1])]

