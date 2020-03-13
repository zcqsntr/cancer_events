import sys
import os
import numpy as np
import tensorflow as tf
import math
import random
from tensorflow import keras



from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras import regularizers
import tensorflow.keras.constraints as cs

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, MaxPooling1D, UpSampling1D
from tensorflow.keras import utils as np_utils
from tensorflow.keras import models
import numpy.random as rng

def data_generator(ndat,nL,sd):
    data = np.zeros((ndat,nL))
    labs = np.zeros((ndat,))
    for i in range(0,ndat):
        # choose dup or del
        evnt = rng.choice([-1,1])
        #labs[i,] = int( (evnt+1)/2 )
        if evnt == -1:
            labs[i,] = 0
        else:
            labs[i,] = 1

        # choose start
        start = rng.randint(0,nL-1)
        #print(start,"\t",type)
        #start = 4

        # create event and add noise
        data[i,start:(start+2)] = evnt
        data[i] += rng.normal(0,sd,(nL))

        #print(data[i,])

    return [data, labs]

def data_generator_multiple(ndat,nL,sd):
    data = np.zeros((ndat,nL))
    labs = np.zeros((ndat,2))

    for i in range(0,ndat):
        start = 0 # so they arent on top of each other

        for j in range(10):
            # choose dup or del
            evnt = rng.choice([-1,1])
            #labs[i,] = int( (evnt+1)/2 )
            if evnt == -1:
                labs[i,0] += 1
            else:
                labs[i,1] += 1

            # choose start
            start = rng.randint(start,start + nL//10-1)
            #print(start,"\t",type)
            #start = 4

            # create event and add noise
            data[i,start:(start+2)] = evnt

            start = start + 2 # move start along
        #data[i] += rng.normal(0,sd,(nL))


        #print(data[i,])
    print(data)
    return [data, labs]



def logistic_regression():
    data, labs = data_generator(ndat, nL, sd)
    labs = np_utils.to_categorical(labs, 2)

    model = Sequential()
    model.add(Dense(units=2, input_dim=nL, activation='softmax'))
    batch_size = 128
    nb_epoch = 50

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(data[:n_train], labs[:n_train], batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
    score = model.evaluate(data[n_train:], labs[n_train:], verbose=0)
    #print('Test score:', score[0])
    print('Test accuracy:', score[1])


def convolution():
    data, labs = data_generator(ndat, nL, sd)
    data = data.reshape((ndat, nL,1))
    print(data[:n_train].shape)
    labs = np_utils.to_categorical(labs, 2)

    #create model
    model = Sequential()#add model layers
    model.add(Conv1D(filters = 2,kernel_size=3, activation='relu', input_shape=(10,1)))
    #model.add(Conv1D(5, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(data[:n_train], labs[:n_train], epochs=50)

    score = model.evaluate(data[n_train:], labs[n_train:], verbose=0)
    print('Test accuracy:', score[1])

    layer_outputs = [layer.output for layer in model.layers]

    activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

    return model, activation_model

def auto_encoder():
    data, labs = data_generator_multiple(ndat, nL, sd)
    data = data.reshape((ndat, nL,1))
    print(data[:n_train].shape)
    #labs = np_utils.to_categorical(labs, 2)

    #create model
    input = Input(shape = (nL,1))
    x = Conv1D(filters = 2, kernel_size=4, activation='relu', padding = 'same')(input)

    encoded = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x) # stride is compression ratio

    x = Conv1D(filters = 2, kernel_size=4, activation='relu', padding = 'same')(encoded)

    x = UpSampling1D(size = 2)(x)

    decoded = Conv1D(filters = 1, kernel_size=4, activation='linear', padding = 'same')(x)

    model = Model(input, decoded)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(data[:n_train], data[:n_train], batch_size = 32,epochs=30)

    score = model.evaluate(data[n_train:], data[n_train:], verbose=0)
    print('Test accuracy:', score[1])

    layer_outputs = [layer.output for layer in model.layers]

    activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

    return model, activation_model

def convolution_multiple():
    data, labs = data_generator_multiple(ndat, nL, sd)
    data = data.reshape((ndat, nL,1))
    print(data[:n_train].shape)
    #labs = np_utils.to_categorical(labs, 2)

    #create model
    model = Sequential()#add model layers
    model.add(Conv1D(filters = 2,kernel_size=4, activation='relu', input_shape=(nL,1)))
    #model.add(Conv1D(5, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(data[:n_train], labs[:n_train], epochs=100)

    score = model.evaluate(data[n_train:], labs[n_train:], verbose=0)
    print('Test accuracy:', score[1])

    layer_outputs = [layer.output for layer in model.layers]

    activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

    return model, activation_model

def plot_filters(model):
    for layer in model.layers:
    	# check for convolutional layer
    	if 'conv' not in layer.name:
    		continue
    	# get filter weights
    	filters, biases = layer.get_weights()

    # retrieve weights from the second hidden layer
    filters, biases = model.layers[0].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()

    #filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = filters.shape[2], 1

    for i in range(n_filters):
        # get the filter
        f = filters[:, :, i]
        # plot each channel separately

        # specify subplot and turn of axis
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f.reshape(len(f),1), cmap='plasma')
        plt.colorbar()
        ix+=1
    # show the figure

    plt.show()


if __name__ == '__main__':
    n_train = 10000
    n_test = 1000

    ndat = n_train + n_test
    nL = 100 #length
    sd = 0.5 #sd of noise


    print()
    print('-----------------Logistic regression---------------------')
    print()
    #logistic_regression()

    print()
    print('-----------------Convolution---------------------')
    print()



    '''
    model, activation_model = convolution()
    plot_filters(model)
    # make a dup and del

    dup = np.zeros((1, nL,1), dtype = np.float32)
    dup[0,1:3,0] = 1
    dup[0] += rng.normal(0,sd,(nL,1))

    dl = np.zeros((1, nL,1), dtype = np.float32)
    dl[0,6:8,0] = -1
    dl[0] += rng.normal(0,sd,(nL,1))

    print(dup)
    print(dl)

    print(activation_model.predict(dup)[0])

    print(activation_model.predict(dl)[0])

    for layer in activation_model.predict(dl):
        print(layer.shape)



    print()
    print('-----------------Convolution multiple events---------------------')
    print()
    model, activation_model = convolution_multiple()

    data, labs = data_generator_multiple(ndat, nL, sd)

    for i in range(4):
        print()
        print(data[i])
        print(model.predict(data[i].reshape((1,nL,1))))
        print(labs[i])
        print(activation_model.predict(data[i].reshape((1,nL,1)))[0])
    '''

    print()
    print('------------------------------Autoencoder---------------------------')
    print()

    model, act_model = auto_encoder()

    data, labs = data_generator_multiple(ndat, nL, sd)

    for i in range(4):
        print()
        print('input:', data[i])
        print('output: ', np.round(model.predict(data[i].reshape((1,nL,1))).reshape(data[i].shape)))
