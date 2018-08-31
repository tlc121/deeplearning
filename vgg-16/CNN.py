import numpy as np
import os
import matplotlib.pyplot as plt
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, Flatten
from keras.optimizers import sgd, rmsprop, adam
from keras.datasets import fashion_mnist, mnist
from keras import Sequential
from keras.utils import np_utils

def normalize(img):
    maxi = np.max(img)
    mini = np.min(img)
    img = (img.astype('float32')-mini)/(maxi-mini)
    return img


(train_input, train_output), (test_input, test_output) = fashion_mnist.load_data()
train_input = train_input.astype('float32')
test_input = test_input.astype('float32')
model = Sequential()

#transfer labels into one-hot
train_output = np_utils.to_categorical(train_output, num_classes=10)
test_output = np_utils.to_categorical(test_output, num_classes=10)

#normalize all images
for i in range(train_input.shape[0]):
    train_input[i, :, :] = normalize(train_input[i])

for i in range(test_input.shape[0]):
    test_input[i, :, :] = normalize(test_input[i])

train_input = train_input[:,:,:,np.newaxis]
test_input = test_input[:,:,:,np.newaxis]


model.add(Convolution2D(input_shape=(28,28,1), filters=32, kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(filters=32, kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=2))
model.add(Flatten())
model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

SGD = sgd(lr = 0.1, decay=1e-6)

model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=SGD)

print ('-----Training-----')
model.fit(train_input, train_output, epochs=2, batch_size=32)


print ('-----Test-----')
loss, accuracy = model.evaluate(test_input, test_output, batch_size=32)
print (loss, accuracy)
