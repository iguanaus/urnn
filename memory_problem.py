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
#allow bigger graphs (Python's default setting of 1000 for modern computers [https://github.com/Theano/Theano/issues/689]):
import sys 
sys.setrecursionlimit(50000)

numcategories = 8#This is the total number of categories (not including the +2). Interesting that they are 2 higher. Hrm......
#Wait just kidding so a 9 would have just been 1,2,3,4,5,6,7,8 (8 categories)

#The reason this has to be +1 is because python truncates the highest one. 
def generate_data(time_steps, n_data, n_sequence):
    seq = np.random.randint(1, high=(numcategories+1), size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps-1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = (numcategories+1) * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    
    return x.T, y.T

    
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, model, input_type, out_every_t, loss_function, w_impl='urnn',n_reflections=None,flag_telescope=True,flag_useGivensForLoop=False,num_cats=8):
    numcategories = int(num_cats)

    # --- Set data params ----------------
    n_input = 10
    n_output = (numcategories+1)
    n_sequence = 10
    n_train = int(1e5)
    n_test = int(1e4)
    num_batches = int(n_train / n_batch)
  

    # --- Create data --------------------
    train_x, train_y = generate_data(time_steps, n_train, n_sequence)
    test_x, test_y = generate_data(time_steps, n_test, n_sequence)

    # train_x is size 120 x n_train, train_y is size 120 x n_train

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)

    
    # --- Create theano graph and compute gradients ----------------------

    print 'Creating theano graph for model %s' % model   
 
    gradient_clipping = np.float32(1)


    if (model == 'LSTM'):           
        inputs, parameters, costs = LSTM(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        print inputs
        for ele in inputs:
            print "Input element:"
            print ele
            #print ele.get_shape()
        print "Donedone"

        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'complex_RNN'):
        if (w_impl == 'full'):
            inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,out_every_t=out_every_t, loss_function=loss_function,output_type='real', fidx=None, flag_return_lin_output=False,name_suffix='',x_spec=None,flag_feed_forward=False,flag_use_mask=False,hidden_bias_mean=0.0,lam=0.0,Wimpl='full',prng_Givens=np.random.RandomState(52016))

        else:
            inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,
                                                out_every_t=out_every_t, loss_function=loss_function)
        #inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,
        #                                        out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'cue_RNN'):
            #print "Using CUE-RNN with %d telescoping Householder reflections of dim. %d" % (n_reflections,n_hidden)
        if flag_telescope:
            print "Using CUE-RNN with %d telescoping Householder reflections of dim. %d" % (n_reflections,n_hidden)
        else:
            print "Using CUE-RNN with %d full Householder reflections of dim. %d" % (n_reflections,n_hidden)
        inputs, parameters, costs = cue_RNN(n_input, n_hidden, n_output, input_type=input_type,
                                                out_every_t=out_every_t, loss_function=loss_function, n_reflections=n_reflections, flag_telescope=True)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'Givens_RNN'):
        inputs, parameters, costs = Givens_RNN(n_input, n_hidden, n_output, input_type=input_type,
                                               out_every_t=out_every_t, loss_function=loss_function,flag_useGivensForLoop=flag_useGivensForLoop)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'RNN'):
        inputs, parameters, costs = tanhRNN(n_input, n_hidden, n_output, input_type=input_type,
                                            out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
    
    else:
        print "Unsuported model:", model
        return

    #Print out the size of things
    for param in parameters:
        print param
        print param.get_value().shape


 
    # --- Compile theano functions --------------------------------------------------

    print 'Compiling theano functions...'

    index = T.iscalar('i')

    if (w_impl == 'full'):
        idx_project=[5]
    else:
        idx_project=None
    updates, rmsprop = rms_prop(learning_rate, parameters, gradients,idx_project)
    #updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1)],
              inputs[1] : s_train_y[:, n_batch * index : n_batch * (index + 1)]}
    print "My givens test"

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    print givens_test
    for key, value in givens_test.iteritems():
        print "Key:" , key , "value:", value, value.get_value(), value.get_value().shape

    #print "my inputs:"
    #for value in inputs:
    #    print "value:", value, value.get_shape(), value.get_value(), value.get_value().shape


   
    train = theano.function([index], costs[0], givens=givens, updates=updates)
    test = theano.function([], [costs[0], costs[1]], givens=givens_test)

    # --- Training Loop ---------------------------------------------------------------

    print 'Starting training loop...'

    train_loss = []
    test_loss = []
    test_acc = []
    best_params = [p.get_value() for p in parameters]
    best_rms = [r.get_value() for r in rmsprop]
    best_test_loss = 1e6
    for i in xrange(n_iter):
        if (n_iter % num_batches == 0):
            inds = np.random.permutation(n_train)
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[:,inds])

        ce = train(i % num_batches)
        train_loss.append(ce)
        print "Iteration:", i
        print "cross entropy:", ce
        print

        if (i % 50==0):
            ce, acc = test()
            print
            print "TEST"
            print "cross entropy:", ce
            print 
            test_loss.append(ce)
            test_acc.append(acc)

            if ce < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_rms = [r.get_value() for r in rmsprop]
                best_test_loss = ce

            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'test_acc': test_acc,
                         'best_params': best_params,
                         'best_rms': best_rms,
                         'best_test_loss': best_test_loss,
                         'model': model,
                         'time_steps': time_steps}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

            
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--n_batch", type=int, default=128)
    parser.add_argument("--n_hidden", type=int, default=40)
    parser.add_argument("--time_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--savefile", default = "oldtest_mem_prob")
    parser.add_argument("--model", default='LSTM')#complex_RNN')
    parser.add_argument("--input_type", default='categorical')
    parser.add_argument("--out_every_t", default='True')
    parser.add_argument("--loss_function", default='CE')
    parser.add_argument("--n_reflections", default=8, help="number of reflections for CUE-RNN")
    parser.add_argument("--flag_telescope", default=False, help="whether to use telescoping reflections (True) or full reflections (False)")
    parser.add_argument("--flag_useGivensForLoop",default=False, help="if True, use a for loop instead of scan to do Givens rotations")
    parser.add_argument("--w_impl", default='urnn_adhoc')
    parser.add_argument("--num_cats",default=8)

    args = parser.parse_args()
    dict = vars(args)

    kwargs = {'n_iter': dict['n_iter'],
              'n_batch': dict['n_batch'],
              'n_hidden': dict['n_hidden'],
              'time_steps': dict['time_steps'],
              'learning_rate': np.float32(dict['learning_rate']),
              'savefile': dict['savefile'],
              'model': dict['model'],
              'input_type': dict['input_type'],
              'out_every_t': 'True'==dict['out_every_t'],
              'loss_function': dict['loss_function'],
              'n_reflections': int(args.n_reflections),
              'flag_telescope': bool(args.flag_telescope),
              'flag_useGivensForLoop': bool(args.flag_useGivensForLoop),
              'w_impl': dict['w_impl'],
              'num_cats': dict['num_cats']}

    main(**kwargs)
