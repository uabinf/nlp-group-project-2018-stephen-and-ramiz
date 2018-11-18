from builtins import range
import numpy as np


def fc_forward(x, w, b):
    out = None
    x_data = x.reshape(x.shape[0], -1)
    out = np.dot(x_data,w) + b
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    x_data = x.reshape(x.shape[0], -1)
    dx = np.reshape(np.dot(dout, w.T), x.shape)
    dw = np.dot(x_data.T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = None
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout
    dx[x < 0] = 0
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        norm = (x - mean) / np.sqrt(var + eps)
        out = gamma * norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        
        cache = (norm,gamma,(x-mean),(1/np.sqrt(var+eps)),np.sqrt(var+eps),var,eps)
    elif mode == 'test':
        norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    
    norm,gamma,xmu,ivar,sqrtvar,var,eps = cache
    
    dgamma = np.sum(dout * norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # Reference: code obtained from https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    N,D = dout.shape

    dxhat = dout * gamma

    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar

    dsqrtvar = -1. /(sqrtvar**2) * divar

    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

    dsq = 1. /N * np.ones((N,D)) * dvar

    dxmu2 = 2 * xmu * dsq

    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

    dx2 = 1. /N * np.ones((N,D)) * dmu

    dx = dx1 + dx2

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)

    N, D = x.shape
    
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * norm + beta
        
    cache = (norm,gamma,(x-mean),(1/np.sqrt(var+eps)),np.sqrt(var+eps),var,eps)
    
    return out, cache


def layernorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    
    norm,gamma,xmu,ivar,sqrtvar,var,eps = cache
    
    dgamma = np.sum(dout * norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # Reference: code obtained from https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    N,D = dout.shape

    dxhat = dout * gamma

    divar = np.sum(dxhat*xmu, axis=1, keepdims=1)
    dxmu1 = dxhat * ivar

    dsqrtvar = -1. /(sqrtvar**2) * divar

    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

    dsq = 1. /D * np.ones((N,D)) * dvar

    dxmu2 = 2 * xmu * dsq

    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=1, keepdims=1)

    dx2 = 1. /D * np.ones((N,D)) * dmu

    dx = dx1 + dx2
    
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
