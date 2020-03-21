import numpy as np

class FullyConnected:
    #fully connected layer using softmax as activation function

    def __init__(self, length, nodes):
        self.weights= np.random.randn(length, nodes)/length
        self.biases= np.zeros(nodes)

    def forward(self, data):
        #forward pass of FC layer over the data given as output
        #from maxpool layer and returns a 1d array with the probability values
        self.latestDataShape=data.shape #save to reconstruct in backprop
        #flatten 
        flattened_shape = data.shape[:1] + (-1,)
        data= np.reshape(data, flattened_shape)
        #linear
        self.latestData=data #store for backprop
        total= np.dot(data, self.weights)+self.biases
        #softmax
        self.latestTotal =total #store for backprop
        x = total - np.max(total, axis=1, keepdims=True)
        exp_x = np.exp(x)
        s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return s
        
    
    def Backprop(self,pre_grad,learn_rate):
        der = np.ones(self.latestTotal.shape)
        act_grad = pre_grad*der
        dw = np.dot(self.latestData.T, act_grad)
        db = np.mean(act_grad, axis=0)
        self.weights -= learn_rate * dw
        self.biases -= learn_rate * db
        #unflatten
        return np.reshape(np.dot(act_grad,self.weights.T),self.latestDataShape)