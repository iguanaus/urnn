import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from optimizations import *    
import argparse, timeit

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
        
def generate_data(time_steps, n_data, n_sequence):
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps-1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    
    return x.T, y.T

    
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, model, input_type, out_every_t, loss_function):
     config={'learning_rate' : 1e-4,
            'learning_rate_natGrad' : None,
            'clipnorm' : 1.0,
            'batch_size' : 128,
            'nb_epochs' : 200,
            'patience' : 30,
            'hidden_units' : 40, #
            'model_impl' : 'complex_RNN',
            'unitary_impl' : 'ASB2016',
            'histfile' : 'exp/history_mnist_default',
            'savefile' : 'exp/model_mnist_default.hdf5',
            'savefile_init' : None}

    # --- Set data params ----------------
    nb_classes = 10
    n_output = 9
    n_sequence = 10
    n_train = int(1e5)
    n_test = int(1e4)
    num_batches = int(n_train / n_batch)

    clipnorm = config['clipnorm']
    batch_size = config['batch_size']
    nb_epochs = config['nb_epochs']
    hidden_units = config['hidden_units']
    # ASB2016 uRNN has 32N+10 parameters
    # full uRNN has N^2+25N+10 parameters

    #model_impl='uRNN_keras'
    #model_impl='complex_RNN'
    model_impl=config['model_impl']
    unitary_impl=config['unitary_impl']

    histfile=config['histfile']
    savefile=config['savefile']
    learning_rate = config['learning_rate']
    if ('learning_rate_natGrad' in config) and (config['learning_rate_natGrad'] is not None):
        learning_rate_natGrad = config['learning_rate_natGrad']
    else:
        learning_rate_natGrad = learning_rate

  

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
        model.add(LSTM(hidden_units,
                       return_sequences=False,
                       input_shape=s_train_x.shape[1:]))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))     

    rmsprop = RMSprop_and_natGrad(lr=learning_rate,clipnorm=clipnorm,lr_natGrad=learning_rate_natGrad)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    history=LossHistory(histfile)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=savefile, verbose=1, save_best_only=True)
    earlystopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], verbose=1, mode='auto') 

    if not (config['savefile_init'] is None):
        print("Loading weights from file %s" % config['savefile_init'])
        model.load_weights(config['savefile_init'])
        losses = model.test_on_batch(X_valid,Y_valid)
        print("On validation set, loaded model achieves loss %f and acc %f"%(losses[0],losses[1]))

    #make sure the experiment directory to hold results exists
    if not os.path.exists('exp'):
        os.makedirs('exp')

    model.fit(s_train_x, s_train_y, batch_size=batch_size, nb_epoch=nb_epochs,
              verbose=1, callbacks=[history,checkpointer,earlystopping])

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # add test scores to history
    history_load=cPickle.load(open(histfile,'rb'))
    history_load.update({'test_loss' : scores[0], 'test_acc' : scores[1]})
    cPickle.dump(history_load, open(histfile, 'wb')) 





            
if __name__=="__main__":
    main(sys.argv[1:])
