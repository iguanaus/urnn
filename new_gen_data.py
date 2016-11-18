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
import random

def generate_data(time_steps, n_data, n_sequence,category_size=10):
    seq = np.random.randint(1, high=category_size-1, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps-1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = (category_size-1) * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    
    return x.T, y.T

#Okay so I am actually going to use a psuedo-code of their method instead. 

def oneSequence(ind, T, input_len, category_size):
    #print(ind,T,input_len,category_size)
    input_list = []
    temp = ind
    for j in range(input_len):
        input_digit = [0] * (category_size + 2) #Add 2 to category size, then create input_digit that is 0's of that length
        input_digit[temp%category_size] = 1 #temp mod. make the one with temp = 1

        temp = temp/category_size
        input_list.append(input_digit)
    #print(input_list)
    
    blank_digit = [0] * (category_size + 2) #This is the blankt
    blank_digit[category_size] = 1
    # print blank_element
    wait_memory = [blank_digit] * (T - 1)
    wait_output = [blank_digit] * input_len

    signal_digit = [0] * (category_size + 2) #This is the signal afterward
    signal_digit[category_size + 1] = 1
    signal = [signal_digit]

    input_element = np.array(input_list + wait_memory + signal + wait_output)

    blank_output = [blank_digit] * (input_len + T)
    output_element = np.array(blank_output + input_list)

    #print("Input:")
    #print(input_element)

    #print("Output:")
    #print(output_element)
    
    return input_element, output_element

def copyingProblemData(training_size,test_size,T,input_len,cat_size):
    #Okay so in this I want a random integer between 1 and a higher number, but with no repeats. I will make a class for this, and then just store values in a dict. 
    N = training_data_size/(category_size ** input_len)
    numCountLimit = N+1
    #os.exit()
    alreadyNums = {}
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    print "Limits created, while loop started.."
    myupperval = cat_size**input_len
    while True:
        myint = random.randint(1,myupperval)
        val = alreadyNums.get(myint,0)
        #print "Storing val..",val
        if val == 0:
        	alreadyNums[myint] = 1
        else:
        	alreadyNums[myint] += 1
	        #alreadyNums[myint] += alreadyNums.get(myint,0)
        if (alreadyNums[myint]<=numCountLimit):
            input_element, output_element = \
                 oneSequence(myint, T, input_len, cat_size)
            #print ("Input")
            #print(input_element)
            #print(output_element)
            if len(train_input)<training_size:
                print("Filling test....", len(train_input)*1.0/training_size)
                train_input.append(input_element)
                train_output.append(output_element)
            elif len(test_input)<test_size:
                print("Filling test....", len(test_input)*1.0/test_size)
                test_input.append(input_element)
                test_output.append(output_element)
            else:
                break

        else:
            continue

    data = {'train': {'Input': train_input, 'Output': train_output}, \
            'test': {'Input': test_input, 'Output': test_output}}

    data_param = {
        'training_data_size': training_data_size,\
        'testing_data_size': test_size,\
        'T': T, \
        'input_len': input_len,\
        'category_size': cat_size
    }

    return data, data_param

    





if __name__ == "__main__":
    #test
    print('Running this will generate an example data set.')
    training_data_size = 100000
    testing_data_size =  10000
    T = 100
    input_len = 10
    category_size = 8
    data_set, data_param = copyingProblemData(training_data_size,testing_data_size,T,input_len,category_size)
    # data_set, data_param = copyingProblemData(100000, 1000, 100, 6, 8)
    print('Training Data:')
    print(np.array(data_set['train']['Input']).shape)
    print(np.array(data_set['train']['Output']).shape)
    #First element is the train
    #print('Testing Data:')
    #print(data_set['test']['Input'])
    #print(data_set['test']['Output'])



















