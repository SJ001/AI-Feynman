# SAve a file with 2*(n-1) columns contaning the (n-1) independent variables and the (n-1) gradients of the trained NN with respect these variables

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .logging import log_exception

is_cuda = torch.cuda.is_available()


def evaluate_derivatives(XY, model, logger=None):
    try:
        pts = XY[:, 0:-1]
        data = XY[:, 0:-1]
        pts = torch.tensor(pts)
        pts = pts.clone().detach()
        is_cuda = torch.cuda.is_available()
        grad_weights = torch.ones(pts.shape[0], 1)
        if is_cuda:
            pts = pts.float().cuda()
            model = model.cuda()
            grad_weights = grad_weights.cuda()
        else:
            pts = pts.float()

        pts.requires_grad_(True)
        outs = model(pts)
        grad = torch.autograd.grad(outs, pts, grad_outputs=grad_weights, create_graph=True)[0]
        save_grads = grad.detach().data.cpu().numpy()
        save_data = np.column_stack((data,save_grads))
        return 1, save_data
    except Exception as e:
        log_exception(logger, e)
        return 0, np.array([[]])
        
