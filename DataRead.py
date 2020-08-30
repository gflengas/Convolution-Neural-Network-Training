import numpy as np
import matplotlib.pyplot as plt
def load_mnist_train():

    labels_path = "train-labels.idx1-ubyte"
    images_path = "train-images.idx3-ubyte"
    
    with open(labels_path,'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    
    with open(images_path,'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28,28)

    return images, labels

def load_mnist_test():

    labels_path = "t10k-labels.idx1-ubyte"
    images_path = "t10k-images.idx3-ubyte"
    
    with open(labels_path,'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    
    with open(images_path,'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28,28)

    return images, labels

def SamplePrint(data,labels):
    for i in range(9):
	    # define subplot
	    plt.subplot(330 + 1 + i)
	    # plot raw pixel data
	    plt.imshow(data[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()
    return

def one_hot(label):
    classes = np.unique(label)
    one_hot_labels = np.zeros((label.shape[0], classes.size))
    for i, c in enumerate(classes):
        one_hot_labels[label == c, i] = 1
    return one_hot_labels

def plotData(loss_history,AccuracyTrain_history,AccuracyTest_history):
    plt.subplot(3, 1, 1)
    plt.plot(range(len(loss_history)), loss_history, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Iterations ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)

    plt.subplot(3,1,3)
    plt.plot(range(len(AccuracyTest_history)), AccuracyTest_history, 'r', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Iterations ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.subplot(3, 1, 2)
    plt.plot(range(len(AccuracyTrain_history)), AccuracyTrain_history, 'b', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Iterations ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.gcf().set_size_inches(15, 12)
    plt.show()
    return

def plotWeights(Conv,FC):
    plt.subplot(2, 1, 1)
    plt.hist(Conv)
    plt.title('Conv layer Weights')

    plt.subplot(2,1,2)
    plt.hist(FC)
    plt.title('FC layer Weights')
    plt.gcf().set_size_inches(15, 12)
    plt.show()
    return