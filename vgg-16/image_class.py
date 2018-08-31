from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Activation
from keras.optimizers import sgd, rmsprop
import numpy as np
from keras import Sequential
from keras.utils import np_utils
from numpy import reshape


def normalize(img):
    maxi = np.max(img)
    mini = np.min(img)
    img = (img.astype('float32')-mini)/(maxi-mini)
    return img

(train_input, train_output), (test_input, test_output) = mnist.load_data()
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


train_input = train_input.reshape(train_input.shape[0], -1)
test_input = test_input.reshape(test_input.shape[0], -1)


model.add(Dense(units=60, input_dim=28*28))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

opti = sgd(lr = 0.005)

model.compile(optimizer=opti, loss = 'categorical_crossentropy', metrics = ['accuracy'])

print '-----Training-----'
model.fit(train_input, train_output, epochs=10, batch_size=32)


print '-----Testing------'
loss, accuracy = model.evaluate(test_input, test_output)
print loss, accuracy