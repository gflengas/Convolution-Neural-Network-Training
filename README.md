# Convolution-Neural-Network-Training
  From scratch Numpy based CNN Training Algorithm developed as part of my Thesis.

## Model description

 &nbsp;&nbsp;&nbsp;&nbsp;The current network for CNN training with Mini-Batch Gradient Descent is developed onpure python using the NumPy library and trained with the MNIST data set. It consists of a Convlayer with 12 3x3 filters initialized with He initialization using Relu as activation function, a Maxpool layer with pool size 2, and a fully connected layer using Softmax as activation function. To measure the accuracy of the network, the Categorical Cross-Entropy Loss functionis used. The training parameters that were used for the results presented are: batch size= 128, learning rate=0.01, 10000 training samples, and 1000 test samples. 

![network image](https://github.com/gflengas/Convolution-Neural-Network-Training/blob/master/pictuers/pic1.png)

Using the network described a Keras model was implemented so comparisons could be made. The run of that model gave the following results for a 20 epochs test: 

![keras results](https://github.com/gflengas/Convolution-Neural-Network-Training/blob/master/pictuers/pic2.png)

Running the same network on the NumPy based implementation, the results are the following: 

![NumPy results](https://github.com/gflengas/Convolution-Neural-Network-Training/blob/master/pictuers/pic3.png)

 &nbsp;&nbsp;&nbsp;&nbsp;We observe similar behavior between the 2 implementations. While the training is closeto the same results with slightly worst accuracy and loss, the testing results appear to be a bit unstable. It's worth noting that the NumPy version can give a bit better results in case we use batch size 100, but it is common for power of 2 batch sizes to offer better runtime so 128 was the one used to execute this example.
### Conv Layer
 &nbsp;&nbsp;&nbsp;&nbsp;The first layer of the network is a Conv layer with 12 3x3 filters. The input of this layer is four-dimensional tensor with shape [batch, channel, height, width]=[batch, 1, 28, 28] and the output is [batch,12,26,26]. Once the Convolution is completed, the output goes through the ReLu activation function and the result is forwarded to the next layer.<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;On this layer, a convolution is applied to small regions of an image, sampling the values of pixels in this region, and converting it into a single pixel. It is applied to each region of pixels in the image, to produce a new image. The idea is that pixels in the new image incorporate information about the surrounding pixels, thus reflecting how well a feature is represented in that area.<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;Using a 12 3x3 filters means that we will have a total of 108 weights, which alongside with biases, are used for classification during the forward phase and are updated during the backpropagation phase. These weights are initialized using the He initialization. This initialization technique was used to counter the vanishing/exploding weights problem, which resulted in the network’s performance slowly declining to 0. Another positive effect was that since normal distribution was replaced by the He initialization, the network results were improved after the 2nd epoch for around 50 % accuracy to 75%.<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;ReLU is the most commonly used activation function in neural networks, especially in CNNs. ReLU stands for rectified linear unit, it is defined as y=max(0,x). Visually, it looks like the following:<br/>
 
![ReLu](https://github.com/gflengas/Convolution-Neural-Network-Training/blob/master/pictuers/pic4.png)

It’s cheap to compute as there is no complicated math. The model can therefore take less time to train or run. 
### Maxpool Layer
 &nbsp;&nbsp;&nbsp;&nbsp;The convolutional layers aren’t supposed to reduce the size of the image significantly. Instead, they make sure that each pixel reflects its neighbors. This makes it possible to performdownscaling, through pooling, without losing important information.<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;A widespread method to do so is max pooling, in other words using the maximum valuefrom a cluster of neurons at a previous layer. Indeed, max-pooling layers have a size and a width. Unlike convolution layers, they are applied to the 2-dimensional depth slices of the image, so the resulting image is of the same depth, just of a smaller width and height by dividing them by the pool size. The presented network uses pool size 2. As a result, the output of the Maxpool Layer is [batch,12,13,13].
### Fully Connected Layer
 &nbsp;&nbsp;&nbsp;&nbsp;The fully-connected layer is a combination of a flattening layer and a dense layer using Softmax as the activation function. The input is flattened into a feature vector and passed through a network of neurons to predict the output probabilities.<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;The rows are concatenated to form a long feature vector. The feature vector is then passed through the dense layer and is multiplied by the layer’s weights, summed with its biases, and passed through a 10 nodes Softmax, each representing each digit since we are working with MNIST. This layer will have 12*13*13 = 2028 weights connected to each node, soa total of 20280 weights, which are initialized with the He initialization as well.<br/>
 &nbsp;&nbsp;&nbsp;&nbsp;Softmax turns arbitrary real values into probabilities. The math behind it is pretty simple: given some numbers,
1. Raise e to the power of each of those numbers. 
2. Sum up all the exponential. This result is the denominator. 
3. Use each number’s exponential as its numerator. 
<img src="https://latex.codecogs.com/gif.latex?S(x_{i})&space;=&space;{e^{x_{i}}&space;\over&space;\sum_{i=1}^{n}&space;e^{x_{i}}}" title="S(x_{i}) = {e^{x_{i}} \over \sum_{i=1}^{n} e^{x_{i}}}" />
After the softmax transformation is applied, the digit represented by the node with the highestprobability will be the output of the CNN!

### Loss Function
 &nbsp;&nbsp;&nbsp;&nbsp;Once the forward part of the CNN is done executed, we have to check how accurate theresults were. A loss function has to be implemented to compare the results with the label of each picture that went through the network. A common loss function to use when predicting multiple output classes is the Categorical Cross-Entropy Loss function, defined as follows:<br/>
 <img src="https://latex.codecogs.com/gif.latex?H(y,\widehat{y})=\sum_{i}&space;y_i&space;log{1&space;\over&space;\widehat{y}}=&space;-\sum_{i}&space;y_i&space;log\widehat{y}" title="H(y,\widehat{y})=\sum_{i} y_i log{1 \over \widehat{y}}= -\sum_{i} y_i log\widehat{y}" />
 
*ŷ*: CNN’s prediction, *y*: the desired output label.<br/>
Since we are working with batches, we need to make predictions over multiple examples,  so the average of the loss over all examples needs to be calculated. 

### Backpropagation
 &nbsp;&nbsp;&nbsp;&nbsp; When we use a feed-forward neural network to accept an input x and produce an output ŷ, information flows forward through the network. The inputs x provide the initial information that then propagates up to the hidden units at each layer and finally produces ŷ. This is called forward propagation. During training, forward propagation can continue onward until it produces a scalar cost J(θ). The back-propagation algorithm, often simply called backprop, allows the information from the cost to then flow backward through the network, tocompute the gradient. Computing an analytical expression for the gradients straight forward, but numerically evaluating such an expression can be computationally expensive. The back-propagation algorithm does so using a simple and inexpensive procedure.<br/>
 
![network flow](https://github.com/gflengas/Convolution-Neural-Network-Training/blob/master/pictuers/pic5.png)

 The parameters of the neural network are adjusted according to the following formulae:<br/>
<img src="https://latex.codecogs.com/gif.latex?W^{[l]}={W^{[l]}-adW^{[l]}}&space;\newline&space;b^{[l]}={b^{[l]}-adb^{[l]}}" title="W^{[l]}={W^{[l]}-adW^{[l]}} \newline b^{[l]}={b^{[l]}-adb^{[l]}}" />

*a* represents learning rate, which allows us to control the value of the performed adjustment. A low learning rate can result in a very slow learning network and a high one can result in to not be able to hit the minimum. The parameters *dW* and *db* are calculated using the chain rule, partial derivatives of loss function with respect to *W*, and *b*. The size of dW and db are the same as that of W and b respectively. These variables are calculated following the formulas:<br/>
<img src="https://latex.codecogs.com/gif.latex?dZ^{[l]}={dA^{[l]}*g'{Z^{[l]}}}&space;\newline&space;dW^{[&space;l&space;]}=&space;{dL&space;\over&space;dW^{[&space;l&space;]}}&space;=&space;{1&space;\over&space;{m}}&space;dZ^{[&space;l&space;]}&space;A^{[&space;l-1&space;]T}&space;\newline&space;db^{[&space;l&space;]}=&space;{dL&space;\over&space;db^{[&space;l&space;]}}&space;=&space;{1&space;\over&space;{m}}&space;\sum_{i=1}^{m}&space;{dZ^{[&space;l&space;](i)}}&space;\newline&space;dA^{[&space;l&space;]}=&space;{dL&space;\over&space;dA^{[&space;l&space;-1]}}&space;=&space;W^{[l]T}dZ^{[l]}" title="dZ^{[l]}={dA^{[l]}*g'{Z^{[l]}}} \newline dW^{[ l ]}= {dL \over dW^{[ l ]}} = {1 \over {m}} dZ^{[ l ]} A^{[ l-1 ]T} \newline db^{[ l ]}= {dL \over db^{[ l ]}} = {1 \over {m}} \sum_{i=1}^{m} {dZ^{[ l ](i)}} \newline dA^{[ l ]}= {dL \over dA^{[ l -1]}} = W^{[l]T}dZ^{[l]}" />
* *Z*: output of a layer
* *A*: activation output of the corresponding layer 
* *m*: number of examples from the training set 
* *g’*: derivative of the non-linear activation function

 &nbsp;&nbsp;&nbsp;&nbsp; The initial gradient that is passed back to the FC layer comes from the computation of the loss function. It is multiplied with the derivative of the Softmax function and the result is used to calculate the dW and db of the FC layer along with the gradient that will be passed to the Maxpool layer. Before this gradient is passed backward, it first needs to be unflatten - reshaped  back to its original dimensions.<br/>
  &nbsp;&nbsp;&nbsp;&nbsp; A Maxpool layer can’t be trained because it doesn’t have any weights, but we still need to implement a method for it to calculate gradients. During the forward pass, the Max Pooling layer takes an input volume and halves its width and height dimensions by picking the max values over 2x2 blocks. The backward pass does the opposite: we’ll double the width and height of the loss gradient by assigning each gradient value to where the original max value was in its corresponding 2x2 block. Each gradient value is assigned to where the original max value was, and every other value is zero. The result is being passed backward to the Conv layer.<br/>
  &nbsp;&nbsp;&nbsp;&nbsp;Conv layer is working similarly to the FC layer. The gradient is first multiplied with the derivative of the ReLu function and the result is used to calculate the dW and db of the FC layer along with the gradient that will be passed to the next layer.

### References 
[Geoffrey E. Hinton Alex Krizhevsky Ilya Sutskever.ImageNet Classificationwith Deep Convolutional Neural Networks. 2012.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
[Gradient BasedLearning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
[Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)
