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
import theano
import matplotlib.pyplot as plt

def plot_learning_curve(histfile,label,color='b',flag_plot_train=False,ax=None):
    ax = plt.gca()
    #if ax is None:
        #create a new figure
        #fig,ax=plt.subplots(1, 2, sharex=True,figsize=(10,5))
    
    try:
        history=cPickle.load(open(histfile,'rb'))
    except:
        print ("Can't open file",histfile,"skipping exp:",label)
        #print ("Can't open history file %s, skipping exp %s" % (histfile,label))
        return ax
    
    train_loss=np.asarray(history['train_loss'])
    val_loss=np.asarray(history['val_loss'])
    val_loss=np.concatenate(([5],val_loss))

    xval=np.arange(val_loss.shape[0])

    xtrain = range(0,len(train_loss))
    
    #xtrain=np.linspace(0,np.max(xval),train_loss.shape[0])

    plt.plot(xtrain,train_loss)


    # if flag_plot_train:
    #     ax[0].plot(xtrain,train_loss)#,color=[c for c in matplotlib.colors.ColorConverter.cache[color]],alpha=0.3)
    # ax[0].plot(xval,val_loss,color=color,label=label,linewidth=4)
    # ax[0].set_xlabel('Epoch')
    
    ax.set_ylabel('Loss (cross-entropy)')
    #ax.set_ylim((0.0,1.))
    
    # train_acc=np.asarray(history['train_acc'])
    # val_acc=np.asarray(history['val_acc'])
    # val_acc=np.concatenate(([0.],val_acc))
    
    # imin=np.argmin(val_loss)
    # print ("Best validation acc. for :", label,val_acc[imin])

    #ax[1].plot(xval,val_acc,color=color,label=label,linewidth=4)
    ax.set_xlabel('Training Iteration')
    ax.set_title("Full URNN, 510 paramters, 8 Categories, T = 100")
    #ax.set_ylabel('Accuracy')
    plt.show()
    return ax


print ("Analyzing....")
plot_learning_curve("exp/history_mnist_default","Full URNN, 510 parameters, 8 Categories, T=100",flag_plot_train=True)












