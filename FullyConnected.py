import numpy as np
import Activations as act
class FullyConnected:
    #fully connected layer using softmax as activation function

    def __init__(self, length, nodes):
        self.weights= np.random.randn(length, nodes)*np.sqrt(2/length) #/length
        self.weights=self.weights.astype('float32')
        #self.weights= np.zeros([length, nodes],dtype='float32')
        self.biases= np.zeros(nodes, dtype='float32')
        self.activ = act.Softmax()

    def forward(self, data):
        #forward pass of FC layer over the data given as output
        #from maxpool layer and returns a 1d array with the probability values
        self.latestData=data #store for backprop
        total= np.dot(data, self.weights)+self.biases
        #softmax
        s=self.activ.forward(total)
        return s
        
    
    def Backprop(self,pre_grad):
        act_grad = pre_grad*self.activ.derivative()
        self.dw = np.dot(self.latestData.T, act_grad)
        self.db = np.mean(act_grad, axis=0, dtype='float32')
        grad= np.dot(act_grad,self.weights.T)
        return grad
    
    def getWeights(self):
        return self.weights
    
    def Update(self,learn_rate):
        # Update filters
        self.weights -= learn_rate * self.dw
        self.biases -= learn_rate * self.db
        return
