import numpy as np
class Activation(object):
    
    def __init__(self):
        self.last_forward = None

class Relu(Activation):

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, input):
        self.last_forward = input
        return np.maximum(0.0, input)
    
    def derivative(self, input=None):
        last_forward = input if input else self.last_forward
        res = np.zeros(last_forward.shape, dtype='float32')
        res[last_forward > 0] = 1.
        return res
    
class Softmax(Activation):

    def __init__(self):
        super(Softmax, self).__init__()
    
    def forward(self, input):
        self.last_forward = input
        x = input - np.max(input, axis=1, keepdims=True)
        exp_x = np.exp(x)
        s = exp_x / np.sum(exp_x, axis=1, dtype='float32', keepdims=True)
        return s
    
    def derivative(self, input=None):
        last_forward = input if input else self.last_forward
        return np.ones(last_forward.shape, dtype='float32')