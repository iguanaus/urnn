#!/usr/bin/python
'''Pixel-by-pixel MNIST using a unitary RNN (uRNN)
'''

from __future__ import print_function

import os,sys,getopt
import yaml
import cPickle

import numpy as np
np.random.seed(314159)  # for reproducibility

import keras.callbacks
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM,TimeDistributed
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
from custom_layers import uRNN,complex_RNN_wrapper
from custom_optimizers import RMSprop_and_natGrad


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

    train_input = data_input[0:training_data_size]
    train_output = data_output[0:training_data_size]

    test_input = data_input[-testing_data_size: -1]
    test_output = data_output[-testing_data_size: -1]

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

def main(argv):

    # --- Set data params ----------------
    n_input = 10
    n_output = 9
    n_sequence = 10
    n_train = int(1e5)
    n_test = int(1e4)
    
    n_batch = 5000
    n_iter = 5000
    n_hidden = 40
    time_steps = 50
    learning_rate = 0.001
    savefile = "testing.txt"
    model = "LSTM"
    input_type = 'real'
    out_every_t = True
    loss_function = 'MSE'

    num_batches = int(n_train / n_batch)
    #data, data_param = copyingData.copyingProblemData(training_data_size, testing_data_size, \
    #        T, input_len, category_size)
  

    # --- Create data --------------------
    train_x, train_y = generate_data(time_steps, n_train, n_sequence)
    test_x, test_y = generate_data(time_steps, n_test, n_sequence)

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)

    
    # --- Create theano graph and compute gradients ----------------------

    gradient_clipping = np.float32(1)

    if (model == 'LSTM'):
        model = Sequential()
        model.add(LSTM(n_input, n_hidden, n_output, input_type=input_type,
             out_every_t=out_every_t, loss_function=loss_function))
        #model.add(Dense(nb_classes))


    #Setting up the model
    rmsprop = RMSprop_and_natGrad(lr=learning_rate,clipnorm=clipnorm,lr_natGrad=learning_rate_natGrad)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])
    history=LossHistory(histfile)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=savefile, verbose=1, save_best_only=True)
    earlystopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], verbose=1, mode='auto') 

    #make sure the experiment directory to hold results exists
    if not os.path.exists('exp'):
        os.makedirs('exp')

    print (model.summary())

    #Now for the actual methods. 
    model.fit(s_train_x, s_train_y, batch_size=batch_size, nb_epoch=nb_epochs,
              verbose=1,callbacks=[history,checkpointer,earlystopping])

    scores = model.evaluate(s_train_x, s_train_y, verbose=0)

    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # add test scores to history
    history_load=cPickle.load(open(histfile,'rb'))
    history_load.update({'test_loss' : scores[0], 'test_acc' : scores[1]})
    cPickle.dump(history_load, open(histfile, 'wb'))     

if __name__ == "__main__":
    main(sys.argv[1:])