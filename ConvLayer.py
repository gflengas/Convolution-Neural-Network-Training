import numpy as np
import Activations as act
class Conv:

    def __init__(self,FiltersNum,FilterSize,Stride):
        self.FiltersNum= FiltersNum
        self.FilterSize= FilterSize
        self.Stride=Stride
        #3d array filter (divide by 9 to reduce variance of initial values)
        self.filter =np.random.randn(FiltersNum,FilterSize,FilterSize)*np.sqrt(2/FiltersNum) # /9
        self.filter=self.filter.astype('float16')
        #self.filter =np.zeros([FiltersNum,FilterSize,FilterSize],dtype='float32')
        self.bias = np.zeros(FiltersNum, dtype='float16')
        self.activ = act.Relu()

    def forward(self,input):
        #gets the image input data and performs forward pass to conv layer
        #and returns the 3d array[y,x,FiltersNum]
        self.latestInput=input #store for backprop
        #shapes
        b,_,h,w = input.shape
        filter_h, filter_w = self.FilterSize,self.FilterSize
        new_h, new_w = (h-filter_h) // self.Stride + 1, (w-filter_w) // self.Stride + 1 
        out = np.zeros((b,self.FiltersNum,new_h,new_w), dtype='float16')
        for batch in range(b):
            for f in range(self.FiltersNum):
                for h in range(new_h):
                    for w in range(new_w):
                        patch = input[batch,:,h:(h+filter_h), w:(w+filter_w)]
                        out[batch,f,h,w] = np.sum(patch*self.filter[f], dtype='float16') + self.bias[f]
    
        self.ConvOut= self.activ.forward(out)
        #print(self.ConvOut.shape)
        #print(self.filter.size)
        return self.ConvOut
    
    def Backprop(self, pre_grad):
        self.dW = np.zeros(self.filter.shape, dtype='float16')
        self.db = np.zeros(self.bias.shape, dtype='float16')
        #relu derivative 
        delta = pre_grad * self.activ.derivative()
        batch,d, img_h, img_w = self.latestInput.shape
        filter_h, filter_w = self.FilterSize,self.FilterSize
        for f in range(self.FiltersNum):
            for ds in range(d):
                for h in range(filter_h):
                    for w in range(filter_w):
                        input_window = self.latestInput[:,ds,
                                           h:img_h - filter_h + h + 1:self.Stride,
                                           w:img_w - filter_w + w + 1:self.Stride]
                        delta_window = delta[:,f]
                        self.dW[f,h,w] = np.sum(input_window * delta_window, dtype='float16') / batch    

        for r in np.arange(self.FiltersNum):
            self.db[r] = np.sum(delta[:, r], dtype='float16') / batch

        layer_grads = np.zeros(self.latestInput.shape)
        return layer_grads
        #return None
    def getFilter(self):
        return self.filter
    
    def Update(self,learn_rate):
        # Update filters
        self.filter = self.filter - learn_rate * self.dW
        self.bias = self.bias - learn_rate * self.db
        return