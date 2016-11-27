#This was the first graph that we generated comparing the full-capacity to our GURNN. Inspecting this graph revealed that we had a number of differences in our code that caused serious statistically significant differences in the results.
#Things that are different:

#       They use RMS prop
#       They use Bengio's style of initiatlization

# Full URNN 2 - their full-capacity, but with a 0.01 learning rate
# baseline - baseline for 8 categories (10*log(8)/(T+20))
# Full URNN - their full-capacity, matching learning rate
# baseline - same baseline as above
# LSTM_param_7967.txt - lstm (our result)
# UniversalURNN - universal urnn bengio result.


import cPickle
import gzip
import theano
import pdb
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse 
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
import theano
import matplotlib.pyplot as plt

def plot_learning_curve(histfile,label,color='b',flag_plot_train=False,ax=None,T=1000,moreresults=None):
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
    if moreresults != None:
        try:
            history2=cPickle.load(open(moreresults,'rb'))
        except:
            print ("Can't open file",moreresults,"skipping exp:",label)
            #print ("Can't open history file %s, skipping exp %s" % (histfile,label))
            return ax
        train_loss=np.asarray(history2['train_loss'])
        val_loss=np.asarray(history2['val_loss'])
        val_loss=np.concatenate(([5],val_loss))
        xval=np.arange(val_loss.shape[0])
        xtrain = range(0,len(train_loss))
        plt.plot(xtrain,train_loss,'.')
        print "Plotted it"



    
    train_loss=np.asarray(history['train_loss'])
    #val_loss=np.asarray(history['val_loss'])
    #val_loss=np.concatenate(([5],val_loss))

    xval=np.arange(train_loss.shape[0])

    xtrain = np.array(range(0,len(train_loss)))*128.0
    
    #xtrain=np.linspace(0,np.max(xval),train_loss.shape[0])

    plt.plot(xtrain,train_loss,label=label)
    x = np.arange(0,(int)(len(train_loss)*1.2))*128.0
    def func(x):
        return 10*np.log(8)/(T+20)
    y = [func(i) for i in x]
    #print len(x)
    #print len(y)
    plt.plot(x,y,label='baseline')


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
    ax.set_title(label)
    #ax.set_ylabel('Accuracy')
    
    return ax

def draw_graph_file(input_file):
    xvals = []
    yvals = []
    with open(input_file) as f:
        count = 0
        for line in f:
            if (count > 1):
                if line != "\n":
                    myvals = line.split()
                    #print myvals
                    xvals.append(myvals[0])
                    yvals.append(myvals[1])
            if line=="########\n":
                #print "Starting!"
                count +=1

    plt.plot(xvals,yvals,label=input_file)




print ("Analyzing....")
#plot_learning_curve("exp/history_mnist_default","Full URNN, 510 parameters, 8 Categories, T=100",flag_plot_train=True)
input_file = "LSTM_param_7967.txt"

#plot_learning_curve("urnn_40_100_2","Full URNN, 1600 parameters, 8 Categories, T=100",flag_plot_train=True,T=100)
plot_learning_curve("memory_problem_complex_RNN_full_complex_RNN_nhidden40_t100","Full URNN 2, 1600 parameters, 8 Categories, T=100",flag_plot_train=True,T=100)
plot_learning_curve("mem_complex_2","Full URNN, 1600 parameters, 8 Categories, T=100",flag_plot_train=True,T=100)
draw_graph_file(input_file)
draw_graph_file("URNN_param_3720.txt")
draw_graph_file("UniversalURNN_param_1668.txt")

plt.legend()
plt.show()












