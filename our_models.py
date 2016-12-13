import theano, cPickle
import theano.tensor as T
import numpy as np
from fftconv import cufft, cuifft
from myutils import *

expTry = False

'''
    Assume this does not have the fast designation
        adhoc

'''

def bengio_RNN(n_input,n_hidden,n_output,input_type='real',out_every_t=False,loss_function='CE'):
    #Initialize states
    #This is the matrix from input -> hidden
    V = initialize_matrix(n_input,2*n_hidden,'V')
    print V
    print V.get_value()
    #Matrix from hidden -> out
    U = initialize_matrix(2*n_hidden,n_output,"U")
    print U
    #Now the output bias. No input bias?
    U_bias_values=np.zeros((n_output,1),dtype=theano.config.floatX)
    U_bias = theano.shared(U_bias_values,name='U Bias')

    #Now the hidden state
    hidden_bias_mean=0
    #The hidden_to_hidden matrix is a list of the different weight matrices
    hidden_bias, h_0, hidden_to_hidden_matrix = initialize_complex_RNN_layer(n_hidden,0,"H")

    #Don't actually understand what this is for
    swap_re_im = np.concatenate((np.arange(n_hidden,2*n_hidden),np.arange(n_hidden)))
    print swap_re_im

    theta = hidden_to_hidden_matrix[0]
    reflection = hidden_to_hidden_matrix[1]
    index_permute_long = hidden_to_hidden_matrix[2]

    #This is the set of all params. h_0 is the initial parameters for h_0. 

    parameters = [V,U,hidden_bias,reflection,U_bias,theta, h_0]

    #for i_layer in xrange(2,n_layers+1):
    #    Wvparams = initialize_unitary
    #I am not doing multiple layers right now
    x ,y = initialize_data_nodes(loss_function,input_type,out_every_t)

    #These are structued to be sequences, non-sequences
    def recurrence(x_t,y_t,ymask_t, h_prev, cost_prev, acc_prev, V, hidden_bias, out_bias, U, *argv):

        # h_prev is of size n_batch x n_layers*2*n_hidden
        Wparams = argv[0:3]
        argv = argv[3:]
        print "Testing recurrence..."
        print n_hidden
        #print h_prev.get_value()
        h_prev_layer1 = h_prev[:,0:2*n_hidden]
        hidden_lin_output = times_unitary(h_prev_layer1,n_hidden,swap_re_im,Wparams)
        msg = theano.printing.Print('T')(x_t)

        #if (input_type=='categorical'):
        data_lin_output = V[T.cast(x_t,'int32')]
        #else:
        #    data_lin_output = T.dot(x_t,V)

        lin_output = data_lin_output

        #Non linearity

        modulus = T.sqrt(1e-5+lin_output**2 + lin_output[:, swap_re_im]**2)
        firstval = modulus+T.tile(hidden_bias,[2]).dimshuffle('x',0)
        rescale = T.maximum(firstval,0.)/(modulus+1e-5)
        h_t = lin_output * rescale

        if (out_every_t):
            lin_output = T.dot(h_t,U) + out.bias.dimshuffle('x',0)
            cost_t , acc_t = compute_cost_t(lin_output,loss_function,y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))

        return h_t, cost_t, acc_t , msg 

    h_0_batch = T.tile(h_0,[x.shape[1],1])

        
    non_sequences = [V   , hidden_bias, U_bias, U] + hidden_to_hidden_matrix

    if (out_every_t):
        print "My x: " , x
	print x.shape
        sequences = [x,y,T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)),[x.shape[0],1,1])]
    else:
        sequences = [x,T.tile(theano.shared(np.zeros((1,1),dtype=theano.config.floatX)),[x.shape[0],1,1]),T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)),[x.shape[0],1,1])]

    #outputs_info = [h_0_batch,theano.shared(np.float32(0,0)),theano.shared(np.float32(0.0))]
    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]

    print "Ready for scan:"
    print recurrence
    print sequences
    print non_sequences
    print outputs_info

    print "N Hidden: " , n_hidden
    print "H prev: " , non_sequences[1].get_value()



    [hidden_states,cost_steps,acc_steps] , updates = theano.scan(fn=recurrence,sequences=sequences,non_sequences=non_sequences,outputs_info=outputs_info)

    if (cost_transform=='magTimesPhase'):
        cosPhase=T.cos(lin_output)
        sinPhase=T.sin(lin_output)
        linMag=np.sqrt(10**(x/10.0)-1e-5)
        yest_real=linMag*cosPhase
        yest_imag=linMag*sinPhase
        yest=T.concatenate([yest_real,yest_imag],axis=2)
        mse=(yest-y)**2
        cost_steps=T.mean(mse*ymask[:,:,0].dimshuffle(0,1,'x'),axis=2)
    elif cost_transform is not None:
        # assume that cost_transform is an inverse DFT followed by synthesis windowing
        lin_output_real=lin_output[:,:,:n_output]
        lin_output_imag=lin_output[:,:,n_output:]
        lin_output_sym_real=T.concatenate([lin_output_real,lin_output_real[:,:,n_output-2:0:-1]],axis=2)
        lin_output_sym_imag=T.concatenate([-lin_output_imag,lin_output_imag[:,:,n_output-2:0:-1]],axis=2)
        lin_output_sym=T.concatenate([lin_output_sym_real,lin_output_sym_imag],axis=2)
        yest_xform=T.dot(lin_output_sym,cost_transform)
        # apply synthesis window
        yest_xform=yest_xform*cost_weight.dimshuffle('x','x',0)
        y_real=y[:,:,:n_output]
        y_imag=y[:,:,n_output:]
        y_sym_real=T.concatenate([y_real,y_real[:,:,n_output-2:0:-1]],axis=2)
        y_sym_imag=T.concatenate([-y_imag,y_imag[:,:,n_output-2:0:-1]],axis=2)
        y_sym=T.concatenate([y_sym_real,y_sym_imag],axis=2)
        y_xform=T.dot(y_sym,cost_transform)
        # apply synthesis window
        y_xform=y_xform*cost_weight.dimshuffle('x','x',0)
        mse=(y_xform-yest_xform)**2
        cost_steps=T.mean(mse*ymask[:,:,0].dimshuffle(0,1,'x'),axis=2)
    cost = cost_steps.mean()
    accuracy = acc_steps.mean()

    costs = [cost,accuracy]
    return [x,y],parameters,costs

def complex_RNN(n_input, n_hidden, n_output, input_type='categorical',out_every_t=True,loss_function='CE'):
    n_layers = 1
    lam = 0.0
    name_suffix = ""
    V_init ='rand'
    U_init = 'rand'
    Wimpl = "adhoc"
    hidden_bias_mean = 0.0
    hidden_bias_init = 'rand'
    h_0_init = 'rand'
    W_init = 'rand'
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize input and output parameters: V, U, out_bias0
    
    # input matrix V
    V = initialize_matrix(n_input, 2*n_hidden, 'V'+name_suffix, rng, init=V_init)
    Vn = V 
    # output matrix U
    U = initialize_matrix(2 * n_hidden, n_output, 'U'+name_suffix, rng, init=U_init)
    Un = U

    # output bias out_bias
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias'+name_suffix)
   
    # initialize layer 1 parameters
    hidden_bias, h_0, Wparams = initialize_complex_RNN_layer(n_hidden,Wimpl,rng,hidden_bias_mean,name_suffix=name_suffix,hidden_bias_init=hidden_bias_init,h_0_init=h_0_init,W_init=W_init)
    

    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))
      
    # restricted parameterization of Arjovsky, Shah, and Bengio 2015
    theta = Wparams[0]
    reflection = Wparams[1]
    index_permute_long = Wparams[2]

    parameters = [V, U, hidden_bias, reflection, out_bias, theta, h_0]
    #Wparams = [theta]
    h_0_all_layers = h_0

    # initialize data nodes
    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)

    # define the recurrence used by theano.scan 
    def recurrence(x_t, y_t, ymask_t, h_prev, cost_prev, acc_prev, V, hidden_bias, out_bias, U, *argv):  
    	print "My x: " , x_t
        # h_prev is of size n_batch x n_layers*2*n_hidden

        # strip W parameters off variable arguments list
        Wparams=argv[0:3]
        argv=argv[3:]
        
        Wimpl_in_scan=Wimpl
        # Compute hidden linear transform: W h_{t-1}
        h_prev_layer1 = h_prev[:,0:2*n_hidden]
        hidden_lin_output = times_unitary(h_prev_layer1,n_hidden,swap_re_im,Wparams,Wimpl_in_scan)

        # Compute data linear transform
        # inputs are categorical, so just use them as indices into V
        data_lin_output = V[T.cast(x_t, 'int32')]
            
        # Total linear output        
        lin_output = hidden_lin_output + data_lin_output

        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        #  add a little bit to sqrt argument to ensure stable gradients,
        #  since gradient of sqrt(x) is -0.5/sqrt(x)
        if expTry:
            pass
        else:
            modulus = T.sqrt(1e-5+lin_output**2 + lin_output[:, swap_re_im]**2)
            rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
            h_t = lin_output * rescale
         
        h_t_all_layers = h_t

        # assume we aren't passing any preactivation to compute_cost
        z_t = None

        lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
    
        print "Lin output: " , lin_output.ndim
        cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t, z_t=z_t, lam=lam)
        
        return h_t_all_layers, cost_t, acc_t
    
    # compute hidden states
    #  h_0_batch should be n_utt x n_layers*2*n_hidden, since scan goes over first dimension of x, which is the maximum STFT length in frames
    h_0_batch = T.tile(h_0_all_layers, [x.shape[1], 1])
    
    non_sequences = [Vn   , hidden_bias, out_bias, Un] + Wparams
    
    sequences = [x, y, T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    print "Scanning...." 
    print sequences
    [hidden_states_all_layers, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                      sequences=sequences,
                                                                      non_sequences=non_sequences,
                                                                      outputs_info=outputs_info)
    print "DoneScanning"
    # get hidden states of last layer
    hidden_states = hidden_states_all_layers[:,:,(n_layers-1)*2*n_hidden:]

    lin_output = T.dot(hidden_states, Un) + out_bias.dimshuffle('x',0)
   
    cost = cost_steps.mean()
    accuracy = acc_steps.mean()
   
    costs = [cost, accuracy]
    
    return [x, y], parameters, costs
        
if __name__=="__main__":
    print "Hi"
    #x_t = 
    #y_t = 
    #ymask_t = 
    #h_prev = 
    #cost_prev =
    #acc_prev = 
    #V = 
    #hidden_bias = 
    #out_bias = 
    #U =
    





#bengio_RNN(5,8,2,input_type='real',out_every_t=True,loss_function='CE')




