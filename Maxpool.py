import numpy as np

class MaxPool:
    
    def RegionScanner(self,img):
        #Finds all the possible 2x2 image regions to pool over
        y,x,_= img.shape
        ym= y//2
        xm= x//2
        for i in range(ym):
            for j in range(xm):
                imreg= img[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield imreg, i, j
    
    def forward(self,data):
        #forward pass of maxpool layer over the data given as output
        #from conv layer and returns a 3d array[y / 2, x / 2, FiltersNum]
        self.latestData=data #store for backprop
        b,y,x,FiltersNum=data.shape
        output= np.zeros((b,y//2, x//2, FiltersNum))
        for batch in range(b):
            for imreg, i , j in self.RegionScanner(data[batch]):
                output[batch,i, j] = np.amax(imreg, axis=(0, 1))
        return output
    
    def Backprop(self, pre_grad):
        grad = np.zeros(self.latestData.shape)
        b=self.latestData.shape[0]
        for batch in range(b):
            for im_region, i, j in self.RegionScanner(self.latestData[batch]):
                h, w, f = im_region.shape
                amax = np.amax(im_region, axis=(0, 1))
                for i2 in range(h):
                    for j2 in range(w):
                        for f2 in range(f):
                            # If this pixel was the max value, copy the gradient to it.
                            if im_region[i2, j2, f2] == amax[f2]:
                                grad[batch,i * 2 + i2, j * 2 + j2,f2] = pre_grad[batch,i, j, f2]

        return grad