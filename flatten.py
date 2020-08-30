import numpy as np

class flatten:
    
    def forward(self,input):
        self.last_input = input
        flattened_shape = input.shape[:2 - 1] + (-1,)
        return np.reshape(input, flattened_shape)
    
    def Backprop(self, pre_grad):
        return np.reshape(pre_grad, self.last_input.shape)