import numpy as np

class MaxPool:
    
    def forward(self,input):
        #forward pass of maxpool layer over the data given as output
        #from conv layer and returns a 3d array[y / 2, x / 2, FiltersNum]
        self.latestInput=input #store for backprop
        b,FiltersNum,h,w=input.shape
        output= np.zeros((b,FiltersNum,h//2, w//2), dtype='float16')
        for batch in range(b):
            for f in range(FiltersNum):
                for i in range(h//2):
                    for j in range(w//2):
                        output[batch,f,i,j] = np.amax(input[batch,f, i:i+2, j:j+2])
        return output
    
    def Backprop(self, pre_grad):
        grad = np.zeros(self.latestInput.shape, dtype='float16')
        b,FiltersNum,y,x=self.latestInput.shape
        for batch in range(b):
            for f in range(FiltersNum):
                for i in range(y//2):
                    for j in range(x//2):
                        patch = self.latestInput[batch, f, i:i+2, j:j+2]
                        max_idx = np.unravel_index(patch.argmax(), patch.shape)
                        y_shift, x_shift = i*2+ max_idx[0], j*2+max_idx[1]
                        grad[batch,f,y_shift,x_shift] = pre_grad[batch,f,i,j]  

        return grad