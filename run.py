import numpy as np
from numba import jit, cuda 
import math
import random
import ConvLayer as c
import Maxpool as mp
import FullyConnected as fc
import DataRead as dr
import flatten as flt
import time
from numpy.random import seed
import tensorflow as tf
import matplotlib.pyplot as plt
#seed(1)
#tf.random.set_seed(1)
start_time = time.time()
#read data
train_images, train_labels =dr.load_mnist_train()
permutation = np.random.permutation(len(train_images))
train_images = train_images[permutation]
train_labels = train_labels[permutation]
train_images=train_images[:10000].reshape((-1, 1, 28, 28)) / 255.0
train_labels=train_labels[:10000]


test_images, test_labels = dr.load_mnist_test()
permutation2 = np.random.permutation(len(test_images))
test_images = test_images[permutation2]
test_labels = test_labels[permutation2]
test_images=test_images[:1000].reshape((-1, 1, 28, 28)) / 255.0
test_labels=test_labels[:1000]

train_labels = dr.one_hot(train_labels)
test_labels = dr.one_hot(test_labels)
#network 
conv = c.Conv(12,3,1) # bx28x28x1 -> bx26x26x8
maxpool= mp.MaxPool() # bx26x26x8 -> bx13x13x8
#conv2 = c.Conv(32,3,1) # bx28x28x1 -> bx26x26x8
#maxpool2= mp.MaxPool() # bx26x26x8 -> bx13x13x8
flat=flt.flatten()
softmax= fc.FullyConnected(13*13*12,10) # bx13x13x8 -> bx10
#softmax= fc.FullyConnected(5*5*32,10) # bx13x13x8 -> bx10
#dr.SamplePrint(test_images,test_labels)

#loss and accuracy function (mean square error)
def CrossEntropyLoss(epsilon, out, label):
    out = np.clip(out, epsilon, 1. - epsilon)
    return out-label

def accuracy(outputs, targets):
    y_predicts = np.argmax(outputs, axis=1)
    y_targets = np.argmax(targets, axis=1)
    acc = y_predicts == y_targets
    return np.mean(acc,dtype='float16')

def loss(epsilon, out, label):
    out = np.clip(out, epsilon, 1. - epsilon)
    return np.mean(-np.sum(label * np.log(out), axis=-1,dtype='float16'))
    #return 0.5 * np.mean(np.sum(np.power(out - label, 2), axis=1))

#forwad pass of the network
def forward(img):
    C1out = conv.forward(img)
    M1out = maxpool.forward(C1out)
    #C2out = conv2.forward(M1out)
    #M2out = maxpool2.forward(C2out)
    Fout = flat.forward(M1out)
    out= softmax.forward(Fout)
    return out
#Forward and backward pass of the network 
def train(im, label):
    # Forward
    out = forward(im)
    # Calculate initial gradient
    grad = CrossEntropyLoss(1e-7, out, label)
    # Backprop
    avgGrad = softmax.Backprop(grad)
    avgGrad = flat.Backprop(avgGrad)
    #avgGrad = maxpool2.Backprop(avgGrad)
    #avgGrad = conv2.Backprop(avgGrad)
    avgGrad = maxpool.Backprop(avgGrad)
    avgGrad = conv.Backprop(avgGrad)
    return out

def Update(learning_rate=0.01):
    softmax.Update(learning_rate)
    conv.Update(learning_rate)
    return


#store data for plot
loss_history = []
AccuracyTrain_history = []
AccuracyTest_history = []
epoch=20
batch_size=128
test_batch_size=128
AccuracyPerEpoc =[]
LossPerEpoc = []
TestAccuracyPerEpoc = []
TestLossPerEpoc = []
print("Start of training")
for epoch in range(epoch):
    # Shuffle data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    #store data for epoch averages 
    train_losses, train_predicts = [], []
    test_losses, test_predicts = [], []

    for i in range(train_images.shape[0] // batch_size):
        batch_begin = i * batch_size
        batch_end = batch_begin + batch_size
        img = train_images[batch_begin:batch_end]
        lbl = train_labels[batch_begin:batch_end]

        fwdout = train(img, lbl)
        Update()
        batchLoss=loss(1e-7,fwdout, lbl)
        train_losses.append(batchLoss)
        loss_history.append(batchLoss)
        batch_acc=accuracy(fwdout,lbl)
        #print ("loss: %.2f, acc: %.2f" %(batchLoss,batch_acc))
        AccuracyTrain_history.append(batch_acc)
        train_predicts.append(batch_acc)
        

    meanloss=float(np.mean(train_losses))
    meanacc=float(np.mean(train_predicts))
    print("Epoch %d, train:[loss %.2f, acc %.2f], " % (
                (epoch+1),meanloss ,meanacc ))
    AccuracyPerEpoc.append(meanacc)
    LossPerEpoc.append(meanloss)

    permutation2 = np.random.permutation(len(test_images))
    test_images = test_images[permutation2]
    test_labels = test_labels[permutation2]

    for i in range(test_images.shape[0] // test_batch_size):
        batch_begin = i * test_batch_size
        batch_end = batch_begin + test_batch_size
        img = test_images[batch_begin:batch_end]
        lbl = test_labels[batch_begin:batch_end]
        fwdout = forward(img)
        batchLoss=loss(1e-7,fwdout, lbl)
        test_acc=accuracy(fwdout,lbl)
        test_losses.append(batchLoss)
        AccuracyTest_history.append(test_acc)
        test_predicts.append(test_acc)

    meantestloss = float(np.mean(test_losses))
    meantestacc = float(np.mean(test_acc))
    print("test:[loss %.2f, acc %.2f]. " % (
                meantestloss, meantestacc))
    TestAccuracyPerEpoc.append(meantestacc)
    TestLossPerEpoc.append(meantestloss)


print("--- %.2f mins ---" % ((time.time() - start_time)/60))
#dr.plotData(loss_history,AccuracyTrain_history,AccuracyTest_history)
#dr.plotData(LossPerEpoc,AccuracyPerEpoc,TestAccuracyPerEpoc)
# summarize history for accuracy
plt.subplot(2, 1, 1)
plt.plot(AccuracyPerEpoc)
plt.plot(TestAccuracyPerEpoc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(2, 1, 2)
plt.plot(LossPerEpoc)
plt.plot(TestLossPerEpoc)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
dr.plotWeights(conv.getFilter().flatten(),softmax.getWeights().flatten())    