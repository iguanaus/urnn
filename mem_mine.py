#!/usr/bin/python
'''Pixel-by-pixel MNIST using a unitary RNN (uRNN)
'''

from __future__ import print_function

import os,sys,getopt
import yaml
import cPickle
from random import shuffle
import numpy as np
np.random.seed(314159)  # for reproducibility

import keras.callbacks
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM,TimeDistributed
from keras.layers import SimpleRNN , TimeDistributedDense
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
from custom_layers import uRNN,complex_RNN_wrapper 
from custom_optimizers import RMSprop_and_natGrad
import theano

class LossHistory(keras.callbacks.Callback):
    def __init__(self, histfile):
        self.histfile=histfile
    
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_acc  = []
        self.val_loss   = []
        self.val_acc    = []

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        cPickle.dump({'train_loss' : self.train_loss, 'train_acc' : self.train_acc, 'val_loss': self.val_loss, 'val_acc' : self.val_acc}, open(self.histfile, 'wb'))     

def oneSequence(ind, T, input_len, category_size):
    input_list = []
    temp = ind
    for j in range(input_len):
        input_digit = [0] * (category_size + 2)
        input_digit[temp%category_size] = 1
        temp = temp/category_size
        input_list.append(input_digit)
    
    
    blank_digit = [0] * (category_size + 2)
    blank_digit[category_size] = 1
    # print blank_element
    wait_memory = [blank_digit] * (T - 1)
    wait_output = [blank_digit] * input_len

    signal_digit = [0] * (category_size + 2)
    signal_digit[category_size + 1] = 1
    signal = [signal_digit]

    input_element = np.array(input_list + wait_memory + signal + wait_output)

    blank_output = [blank_digit] * (input_len + T)
    output_element = np.array(blank_output + input_list)
    
    return input_element, output_element

#This constructs the copy problem data
#It returns a train and a test set. 

def copyingProblemData(training_data_size, testing_data_size, \
                        T, input_len, category_size):
    N = training_data_size/(category_size ** input_len)
    shuffle_list = range(category_size ** input_len) * (N+1)
    shuffle(shuffle_list)

    data_input = []
    data_output = []

    for ind in shuffle_list:
        input_element, output_element = \
             oneSequence(ind, T, input_len, category_size)
        data_input.append(input_element)
        data_output.append(output_element)

    train_input = np.array(data_input[0:training_data_size])
    train_output = np.array(data_output[0:training_data_size])

    test_input = np.array(data_input[-testing_data_size: -1])
    test_output = np.array(data_output[-testing_data_size: -1])

    data = {'train': {'Input': train_input, 'Output': train_output}, \
            'test': {'Input': test_input, 'Output': test_output}}

    data_param = {
        'training_data_size': training_data_size,\
        'testing_data_size': testing_data_size,\
        'T': T, \
        'input_len': input_len,\
        'category_size': category_size
    }

    return data, data_param

def generate_data(time_steps, n_data, n_sequence):
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps-1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    
    return x.T, y.T

    
def main(argv):

    # --- Set data params ----------------
    n_input = 10
    n_output = 9
    n_sequence = 10
    n_train = int(1e5)
    n_test = int(1e4)
   

    batch_size = 5
    nb_epochs = 5 
    n_batch = 5000
    n_iter = 5000
    n_hidden = 40
    time_steps = 50
    learning_rate = 0.001
    savefile = "testing.txt"
    model = "complex_RNN"
    input_type = 'real'
    out_every_t = True
    loss_function = 'MSE'
    unitary_impl = "ASB2016" 
    clipnorm = 1
    learning_rate_natGrad = None
    histfile = 'exp/history_mnist_default'
    patience = 10
    train_data_size = 200
    test_data_size = 12
    T = 1
    input_len = 2 
    category_size = 2 
    nb_classes=category_size+2
    num_batches = int(n_train / n_batch)
    #data, data_param = copyingData.copyingProblemData(training_data_size, testing_data_size, \
    #        T, input_len, category_size)
  

    # --- Create data --------------------

    data_set, data_param = copyingProblemData(train_data_size,test_data_size,T,input_len,category_size)

    train_x = data_set['train']['Input']
    train_y = data_set['train']['Output']

    test_x = data_set['test']['Input']
    test_y = data_set['test']['Output']

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)
    print(train_x.shape)
    print("Classes:",nb_classes) 
    # --- Create theano graph and compute gradients ----------------------

    gradient_clipping = np.float32(1)

    if (model=='complex_RNN'):
        model = Sequential()
        model.add(LSTM(n_hidden,return_sequences=True,input_shape=train_x.shape[1:]))
	#model.add(complex_RNN_wrapper(output_dim=nb_classes,
        #                      hidden_dim=n_hidden,
        #                      unitary_impl=unitary_impl,
        #                      input_shape=train_x.shape[1:])) 
        model.add(TimeDistributedDense(nb_classes))

        #Hopefully this will softmax each.
        model.add(Activation('softmax'))


    #Setting up the model
    rmsprop = RMSprop_and_natGrad(lr=learning_rate,clipnorm=clipnorm,lr_natGrad=learning_rate_natGrad)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])
    history=LossHistory(histfile)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=savefile, verbose=1, save_best_only=True)
    earlystopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto') 

    #make sure the experiment directory to hold results exists
    if not os.path.exists('exp'):
        os.makedirs('exp')

    print (model.summary())

    #Now for the actual methods. 
    print ("X:",train_x.shape)
    print ("Y:",train_y.shape)
    model.fit(train_x, train_y, nb_epoch=nb_epochs,verbose=1,batch_size=batch_size,validation_data=(test_x,test_y),callbacks=[history,checkpointer,earlystopping])

    scores = model.evaluate(s_train_x, s_train_y, verbose=0)

    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # add test scores to history
    history_load=cPickle.load(open(histfile,'rb'))
    history_load.update({'test_loss' : scores[0], 'test_acc' : scores[1]})
    cPickle.dump(history_load, open(histfile, 'wb'))     

if __name__ == "__main__":
    main(sys.argv[1:])
