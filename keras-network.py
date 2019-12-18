import keras
from keras.models import Sequential
from keras.layers import Dense

import numpy
import random, math, os

model = Sequential()

model.add(Dense(units=2, activation='tanh', input_dim=2))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=64, activation='tanh'))
model.add(Dense(units=2, activation='tanh'))

model.compile(loss="mse", optimizer="rmsprop")

x_train = []
y_train = []
for i in range(5000):
	r = random.uniform(0, 1000)
	theta = random.uniform(0, 2*math.pi)
	
	x = r*math.cos(theta)
	y = r*math.sin(theta)
	
	x_train.append([r/1000, theta/10])
	y_train.append([x/1000, y/1000])
	
	
x_train = numpy.array(x_train)
y_train = numpy.array(y_train)

if os.path.isfile('weights.h5'):
	model.load_weights('weights.h5')
	
	guess = model.predict(x_train, batch_size=128)
	x_train[0] = x_train[0]*1000
	x_train[1] = x_train[1]*10
	y_train = y_train*1000
	guess = guess*1000
	
	for i in range(10):
		print(x_train[i])
		print(y_train[i], guess[i])
		print('% Error', abs((y_train[i] - guess[i])/y_train[i]*100))
	
else:		  
	model.fit(x_train, y_train, epochs=1000, batch_size=32)
	model.save_weights('weights.h5')

	x_test = x_train[0]
	y_test = y_train[0]
