from keras.optimizers import SGD

from keras.layers import Dense,Activation
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential


x_data=np.linspace(1, 10, 100)
y_data= np.log(x_data) + np.random.normal(0,0.01,x_data.shape)



test_input = np.linspace(1,20,200)
test_output = np.log(test_input) + np.random.normal(0,0.02,test_input.shape)

model=Sequential()
model.add(Dense(units=1000,input_dim=1))
model.add(Activation('relu'))
model.add(Dense(units=1))
model.add(Activation('relu'))


defsgd=SGD(lr=0.001)
model.compile(optimizer=defsgd,loss='mse')


for step in range(100000):
    cost=model.train_on_batch(x_data, y_data)

    if step%500==0:
        print 'cost:',cost


#W,b=model.layers[0].get_weights()



plt.scatter(test_input, test_output)
y_pred = model.predict(test_input)
plt.plot(test_input, y_pred, 'r-',lw=3)
# y_pred_train=model.predict(x_data)
# plt.scatter(x_data, y_data)
# plt.plot(x_data, y_pred_train, 'r-',lw=3)
# plt.scatter(test_input, test_output)
# plt.plot(test_input, y_pred, 'r-',lw=3)
plt.show()