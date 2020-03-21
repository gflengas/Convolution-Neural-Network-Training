import numpy as np
import math
import random
import conv as c
import Maxpool as mp
import FullyConnected as fc
import DataRead as dr
import time

#network 
conv = c.Conv(8) # bx28x28x1 -> bx26x26x8
maxpool= mp.MaxPool() # bx26x26x8 -> bx13x13x8
softmax= fc.FullyConnected(13*13*8,10) # bx13x13x8 -> bx10
#dr.SamplePrint(test_images,test_labels)

#loss and accuracy function
def CrossEntropyLoss(epsilon, out, label):
    out = np.clip(out, epsilon, 1 - epsilon)
    return out-label

def accuracy(outputs, targets):
    y_predicts = np.argmax(outputs, axis=1)
    y_targets = np.argmax(targets, axis=1)
    acc = y_predicts == y_targets
    return np.mean(acc)*100

def loss(epsilon, out, label):
    out = np.clip(out, epsilon, 1 - epsilon)
    return np.mean(-np.sum(label * np.log(out), axis=-1))

#forwad pass of the network
def forward(img):
    out = conv.forward((img/255)-0.5)
    out = maxpool.forward(out)
    out= softmax.forward(out)
    return out
#Forward and backward pass of the network 
def train(im, label, lr=0.005):
    # Forward
    out = forward(im)
    # Calculate initial gradient
    grad = CrossEntropyLoss(1e-11, out, label)
    # Backprop
    avgGrad = softmax.Backprop(grad, lr)
    avgGrad = maxpool.Backprop(avgGrad)
    conv.Backprop(avgGrad, lr)

    return out


def network(batch_size,test_batch_size,epoch):
    start_time = time.time()
    #read data
    train_images, train_labels =dr.load_mnist_train()
    test_images, test_labels = dr.load_mnist_test()
    train_labels = dr.one_hot(train_labels)
    test_labels = dr.one_hot(test_labels)
    #store data for plot
    loss_history = []
    AccuracyTrain_history = []
    AccuracyTest_history = []
    
    for epoch in range(epoch):
        # Shuffle data
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]
        #store data for epoch averages 
        train_losses, train_predicts, train_targets = [], [], []
        test_losses, test_predicts, test_targets = [], [], []
        for i in range(train_images.shape[0] // batch_size):
            batch_begin = i * batch_size
            batch_end = batch_begin + batch_size
            img = train_images[batch_begin:batch_end]
            lbl = train_labels[batch_begin:batch_end]

            fwdout = train(img, lbl)
            batchLoss=loss(1e-11,fwdout, lbl)
            train_losses.append(batchLoss)
            loss_history.append(batchLoss)
            AccuracyTrain_history.append(accuracy(fwdout,lbl))
            train_predicts.extend(fwdout)
            train_targets.extend(lbl)

        print("Epoch %d, train-[loss %.2f, acc %.2f]; " % (
                    (epoch+1), float(np.mean(train_losses)), float(accuracy(train_predicts, train_targets))))

        permutation2 = np.random.permutation(len(test_images))
        test_images = test_images[permutation2]
        test_labels = test_labels[permutation2]

        for i in range(train_images.shape[0] // test_batch_size):
            batch_begin = i * test_batch_size
            batch_end = batch_begin + test_batch_size
            img = test_images[batch_begin:batch_end]
            lbl = test_labels[batch_begin:batch_end]
            fwdout = forward(img)
            batchLoss=loss(1e-11,fwdout, lbl)
            test_losses.append(batchLoss)
            AccuracyTest_history.append(accuracy(fwdout,lbl))
            test_predicts.extend(fwdout)
            test_targets.extend(lbl)

        print("Epoch %d, test-[loss %.2f, acc %.2f]; " % (
                    (epoch+1), float(np.mean(test_losses)), float(accuracy(test_predicts, test_targets))))
    

    print("--- %.2f mins ---" % ((time.time() - start_time)/60))
    dr.plotData(loss_history,AccuracyTrain_history,AccuracyTest_history)
    return

network(32,100,3)