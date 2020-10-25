import numpy as np
import DataRead as dr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD

train_images, train_labels =dr.load_mnist_train()
train_images=train_images[:10000]
train_labels=train_labels[:10000]
test_images, test_labels = dr.load_mnist_test()
test_images=test_images[:1000]
test_labels=test_labels[:1000]

train_images = (train_images / 255)
test_images = (test_images / 255)

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = Sequential([
  Conv2D(12, 3,1, activation='relu',kernel_initializer='lecun_normal', input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=2),
  #Conv2D(8, 3, input_shape=(13, 13, 1), use_bias=False),
  #MaxPooling2D(pool_size=2),
  Flatten(),
  Dense(10, activation='softmax',kernel_initializer='lecun_normal'),
])

model.compile(SGD(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(
  train_images,
  to_categorical(train_labels),
  batch_size=128,
  epochs=20,
  validation_data=(test_images, to_categorical(test_labels)),
)
# summarize history for accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()