import numpy as np
from src.layers import *

class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
                 
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for i in range(self.num_layers):
            if i == 0:
                dimIn, dimOut = input_dim, hidden_dims[i]
            elif i == len(hidden_dims):
                dimIn, dimOut = hidden_dims[i-1], num_classes
            else:
                dimIn, dimOut = hidden_dims[i-1], hidden_dims[i]
                
            Wname = 'W' + str(i+1)
            bname = 'b' + str(i+1)
            
            self.params[Wname] = np.random.normal(0.0, weight_scale, (dimIn, dimOut))
            self.params[bname] = np.zeros(dimOut)
            
            if self.normalization == 'batchnorm' and i != self.num_layers - 1:
                gammaName = 'gamma' + str(i+1)
                betaName = 'beta' + str(i+1)
                self.params[gammaName] = np.ones(dimOut)
                self.params[betaName] = np.zeros(dimOut)
            if self.normalization == 'layernorm' and i != self.num_layers - 1:
                gammaName = 'gamma' + str(i+1)
                betaName = 'beta' + str(i+1)
                self.params[gammaName] = np.ones(dimOut)
                self.params[betaName] = np.zeros(dimOut)

                

        self.dropout_param = {}
        if self.use_dropout:
            self.do_params = [{} for i in range(self.num_layers - 1)]
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.ln_params = [{} for i in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        intermediate_scores = []
        intermediate_inputs = []
        input = X.reshape(X.shape[0], -1)
        intermediate_inputs.append(input)
        for i in range(self.num_layers):
            Wname = 'W' + str(i+1)
            bname = 'b' + str(i+1)
            scores, af_cache = fc_forward(input, self.params[Wname], self.params[bname])
                
            if i != self.num_layers - 1:
                if self.normalization == 'batchnorm':
                    gammaName = 'gamma' + str(i+1)
                    betaName = 'beta' + str(i+1)
                    scores, bn_cache = batchnorm_forward(scores, self.params[gammaName], self.params[betaName], self.bn_params[i])
                    self.bn_params[i]['cache'] = bn_cache
                if self.normalization == 'layernorm':
                    gammaName = 'gamma' + str(i+1)
                    betaName = 'beta' + str(i+1)
                    scores, ln_cache = layernorm_forward(scores, self.params[gammaName], self.params[betaName], self.ln_params[i])
                    self.ln_params[i]['cache'] = ln_cache
                    
                intermediate_scores.append(scores)
                input, relu_cache = relu_forward(scores)
            
                if self.use_dropout:
                    input, self.do_params[i]['cache'] = dropout_forward(input, self.dropout_param)
                
                intermediate_inputs.append(input)
                

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        
        loss, dout = softmax_loss(scores, y)
        reg_sum = 0
        for i in reversed(range(self.num_layers)):
            Wname = 'W' + str(i+1)
            bname = 'b' + str(i+1)
            dout, grads[Wname], grads[bname] = fc_backward(dout, (intermediate_inputs[i],self.params[Wname],self.params[bname]))
            if i != 0:
                if self.use_dropout:
                    dout = dropout_backward(dout, self.do_params[i-1]['cache'])
                    
                dout = relu_backward(dout, intermediate_scores[i-1])
                
                if self.normalization == 'batchnorm':
                    gammaName = 'gamma' + str(i)
                    betaName = 'beta' + str(i)
                    dout, grads[gammaName], grads[betaName] = batchnorm_backward(dout, self.bn_params[i-1]['cache'])
                if self.normalization == 'layernorm':
                    gammaName = 'gamma' + str(i)
                    betaName = 'beta' + str(i)
                    dout, grads[gammaName], grads[betaName] = layernorm_backward(dout, self.ln_params[i-1]['cache'])
            
            grads[Wname] += self.reg * self.params[Wname]
            reg_sum += self.reg * np.sum(self.params[Wname]*self.params[Wname])
        loss += 0.5 * reg_sum
        

        return loss, grads

