import numpy as np

class Conv:

    def __init__(self,FiltersNum):
        self.FiltersNum= FiltersNum
        #3d array filter (divide by 9 to reduce variance of initial values)
        self.filter =np.random.randn(FiltersNum,3,3) /9

    def RegionScanner(self,img):
        #Finds all the possible 3x3 image regions using valid padding
        y,x = img.shape
        for i in range(y-2):
            for j in range(x-2):
                imreg =img[i:(i+3), j:(j+3)]
                yield imreg,i,j
    
    def forward(self,data):
        #gets the image input data and performs forward pass to conv layer
        #and returns the 3d array[y,x,FiltersNum]
        self.latestData=data #store for backprop
        b,y,x = data.shape
        result = np.zeros((b,y - 2, x - 2, self.FiltersNum))
        for batch in range(b):
            for im_region, i, j in self.RegionScanner(data[batch]):
                result[batch,i, j] = np.sum(im_region * self.filter, axis=(1, 2))
        return result
    
    def Backprop(self, pre_grad, learn_rate):
        dW = np.zeros(self.filter.shape)
        b= pre_grad.shape[0]
        for batch in range(b):
            for im_region, i, j in self.RegionScanner(self.latestData[batch]):
                for f in range(self.FiltersNum):
                    dW[f] += pre_grad[batch,i, j, f] * im_region
        dW=dW / b
        # Update filters
        self.filter -= learn_rate * dW
        return None