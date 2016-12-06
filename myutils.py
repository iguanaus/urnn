import theano, cPickle
import theano.tensor as T
import numpy as np
from fftconv import cufft, cuifft


#This initializes a n_in by n_out matrix and sets its name
def initialize_matrix(n_in,n_out,name,rng=None,init='rand'):
    if (init=='rand'):
        sigma = np.sqrt(6. / (n_in+n_out))
        values = np.asarray(np.random.uniform(low=-sigma,high=sigma,size=(n_in,n_out)),dtype=theano.config.floatX)
    return theano.shared(value = values, name=name)

#This creates a unitary matrix based on Bengio's paper. This is just an initialize thingy.
#I assume that this is adhoc
def initialize_unitary(dim_size,impl="",rng=None,name ="H"):
    r_mat=initialize_matrix(2,2*dim_size,name+'_reflection')
    sigma = np.pi#From bengio
    #I don't fully get what this is/why it is what it is. Why is it size 3 X n?
    theta_values = np.asarray(np.random.uniform(low=-sigma,high=sigma,size=(3,dim_size)),dtype=theano.config.floatX)
    theta = theano.shared(theta_values,name=name+"_theta")
    #This just gives a random permutation of the numbers 0-9.
    pi_permute = np.random.permutation(dim_size)
    print theta.get_value()
    print pi_permute
    #This is a permuted matrix that is double length (just increased by the dim size), not totally sure why. 
    pi_permute_long = np.concatenate((pi_permute,pi_permute+dim_size))
    print pi_permute_long
    Wparams=[theta,r_mat,pi_permute_long]
    return Wparams
    #print pi_permute #This just gives you a random 
    pass



def initialize_complex_RNN_layer(n_hidden,Wimpl="",rng=None,hidden_bias_mean=0,name_suffix="H",hidden_bias_init=None,h_0_init=None,W_init=None):
    sigma = 0.01
    #Get the bias of recurrent iterations
    hidden_bias_values = np.asarray(hidden_bias_mean+np.random.uniform(low=-sigma,high=sigma,size=(n_hidden,1)),dtype=theano.config.floatX)
    hidden_bias = theano.shared(hidden_bias_values,name=name+"_bias")
    #Get the main matrix initial state. This will be N parameters, but we will normalize each to 1 so we actually need to generate 2*N values right now.
    hidden_size=(1,2*n_hidden)
    sigma = np.sqrt(3./2/n_hidden) #Bengio method
    h_0 = theano.shared(np.asarray(np.random.uniform(low=-sigma,high=sigma,size=hidden_size),dtype=theano.config.floatX))

    #Now these are the initial states. Note that they are ordered just as one list of size 1x 2*N, so I will have to figure out how to process this/which parts are the imag/real parts

    hidden_to_hidden_matrix = initialize_unitary(n_hidden,name=name)
    return hidden_bias, h_0, hidden_to_hidden_matrix

#This sets up the input nodes. They are just empty placeholders right now.
def initialize_data_nodes(loss_function,input_type,out_every_t):
    x = T.tensor3()
    if 'CE' in loss_function:
        if (out_every_t):
            y = T.matrix(dtype='int32')
        else:
            y = T.vector(dtype='int32')
    return x, y

def times_diag(input, n_hidden, diag, swap_re_im):
    # input is a Ix2n_hidden matrix, where I is number
    # of training examples
    # diag is a n_hidden-dimensional real vector, which creates
    # the 2n_hidden x 2n_hidden complex diagonal matrix using 
    # e.^{j.*diag}=cos(diag)+j.*sin(diag)
    d = T.concatenate([diag, -diag]) #d is 2n_hidden
    
    Re = T.cos(d).dimshuffle('x',0)
    Im = T.sin(d).dimshuffle('x',0)

    input_times_Re = input * Re
    input_times_Im = input * Im

    output = input_times_Re + input_times_Im[:, swap_re_im]
   
    return output
    
    
def vec_permutation(input, index_permute):
    return input[:, index_permute]      

    
def times_reflection(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]
   
    vstarv = (reflection**2).sum()
    
    input_re_reflect_re = T.dot(input_re, reflect_re)
    input_re_reflect_im = T.dot(input_re, reflect_im)
    input_im_reflect_re = T.dot(input_im, reflect_re)
    input_im_reflect_im = T.dot(input_im, reflect_im)

    a = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
    b = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
    c = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
    d = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)
         
    output = input
    output = T.inc_subtensor(output[:, :n_hidden], - 2. / vstarv * (a + b))
    output = T.inc_subtensor(output[:, n_hidden:], - 2. / vstarv * (d - c))

    return output    

def do_ifft(input, n_hidden):
    ifft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    ifft_input = ifft_input.dimshuffle(0,2,1)
    ifft_output = cuifft(ifft_input) / T.sqrt(n_hidden)
    ifft_output = ifft_output.dimshuffle(0,2,1)
    output = T.reshape(ifft_output, (input.shape[0], 2*n_hidden))
    return output

def do_fft(input, n_hidden):
    fft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    fft_input = fft_input.dimshuffle(0,2,1)
    fft_output = cufft(fft_input) / T.sqrt(n_hidden)
    fft_output = fft_output.dimshuffle(0,2,1)
    output = T.reshape(fft_output, (input.shape[0], 2*n_hidden))
    return output

#I will use Bengio's method
def times_unitary(x,n,swap_re_im,Wparams,Wimpl=""):
    theta=Wparams[0]
    reflection=Wparams[1]
    index_permute_long=Wparams[2]
    step1 = times_diag(x, n, theta[0,:], swap_re_im)
    step2 = do_fft(step1, n)
    step3 = times_reflection(step2, n, reflection[0,:])
    step4 = vec_permutation(step3, index_permute_long)
    step5 = times_diag(step4, n, theta[1,:], swap_re_im)
    step6 = do_ifft(step5, n)
    step7 = times_reflection(step6, n, reflection[1,:])
    step8 = times_diag(step7, n, theta[2,:], swap_re_im)     
    y = step8
    return y


#This is averaged?!?!??!?!?!?
def compute_cost_t(lin_output,loss_function,y_t):
    RNN_output = T.nnet.softmax(lin_output)
    CE = T.nnet.categorical_crossentropy(RNN_output,y_t)
    cost_t = CE.mean()
    acc_t = (T.eq(T.argmax(RNN_output, axis=-1),y_t)).mean(dtype=theano.config.floatX)
    return cost_t, acc_t




if __name__=="__main__":
    initialize_unitary(10,"Hidden")

