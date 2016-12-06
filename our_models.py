import theano, cPickle
import theano.tensor as T
import numpy as np
from fftconv import cufft, cuifft
from myutils import *



'''
    Assume this does not have the fast designation
        full_

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

def complex_RNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE', output_type='real', fidx=None, flag_return_lin_output=False,name_suffix='',x_spec=None,flag_feed_forward=False,flag_use_mask=False,hidden_bias_mean=0.0,lam=0.0,Wimpl="adhoc",prng_Givens=np.random.RandomState(),Vnorm=0.0,Unorm=0.0,flag_return_hidden_states=False,n_layers=1,cost_weight=None,cost_transform=None,flag_noComplexConstraint=0,seed=1234,V_init='rand',U_init='rand',W_init='rand',h_0_init='rand',out_bias_init='rand',hidden_bias_init='rand',flag_add_input_to_output=False):

    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Initialize input and output parameters: V, U, out_bias0
    
    # input matrix V
    if flag_noComplexConstraint and (input_type=='complex'):
        V = initialize_matrix(2*n_input, 2*n_hidden, 'V'+name_suffix, rng, init=V_init)
        Vaug = V
    else:
        V = initialize_matrix(n_input, 2*n_hidden, 'V'+name_suffix, rng, init=V_init)
        if (Vnorm>0.0):
            # normalize the rows of V by the L2 norm (note that the variable V here is actually V^T, so we normalize the columns)
            Vr = V[:,:n_hidden]
            Vi = V[:,n_hidden:]
            Vnorms = T.sqrt(1e-5 + T.sum(Vr**2,axis=0,keepdims=True) + T.sum(Vi**2,axis=0,keepdims=True))
            Vn = T.concatenate( [Vr/(1e-5 + Vnorms), Vi/(1e-5 + Vnorms)], axis=1)
            # scale so row norms are desired number
            Vn = V*T.sqrt(Vnorm)
        else:
            Vn = V

        if input_type=='complex':
            Vim = T.concatenate([ (-1)*Vn[:,n_hidden:], Vn[:,:n_hidden] ],axis=1) #concatenate along columns to make [-V_I, V_R]
            Vaug = T.concatenate([ Vn, Vim ],axis=0) #concatenate along rows to make [V_R, V_I; -V_I, V_R]
    

    # output matrix U
    if flag_noComplexConstraint and (input_type=='complex'):
        U = initialize_matrix(2*n_hidden,2*n_output,'U'+name_suffix,rng, init=U_init)
        Uaug=U
    else:
        U = initialize_matrix(2 * n_hidden, n_output, 'U'+name_suffix, rng, init=U_init)
        if (Unorm > 0.0):
            # normalize the cols of U by the L2 norm (note that the variable U here is actually U^H, so we normalize the rows)
            Ur = U[:n_hidden,:]
            Ui = U[n_hidden:,:]
            Unorms = T.sqrt(1e-5 + T.sum(Ur**2,axis=1,keepdims=True) + T.sum(Ui**2,axis=1,keepdims=True))
            Un = T.concatenate([ Ur/(1e-5 + Unorms), Ui/(1e-5 + Unorms) ], axis=0)
            # scale so col norms are desired number
            Un = Un*T.sqrt(Unorm)
        else:
            Un = U

        if output_type=='complex':
            Uim = T.concatenate([ (-1)*Un[n_hidden:,:], Un[:n_hidden,:] ],axis=0) #concatenate along rows to make [-U_I; U_R]
            Uaug = T.concatenate([ Un,Uim ],axis=1) #concatante along cols to make [U_R, -U_I; U_I, U_R]
            # note that this is a little weird compared to the convention elsewhere in this code that
            # right-multiplication real-composite form is [A, B; -B, A]. The weirdness is because of the original
            # implementation, which initialized U for real-valued outputs as U=[A; B], which really should have
            # been U=[A; -B]

    
    # output bias out_bias
    if output_type=='complex':
        out_bias = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX), name='out_bias'+name_suffix)
    else:
        out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias'+name_suffix)
   
    
    # initialize layer 1 parameters
    hidden_bias, h_0, Wparams = initialize_complex_RNN_layer(n_hidden,Wimpl,rng,hidden_bias_mean,name_suffix=name_suffix,hidden_bias_init=hidden_bias_init,h_0_init=h_0_init,W_init=W_init)

    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))
    
    if (Wimpl=='adhoc_fast'):
        # create the full unitary matrix from the restricted parameters,
        # since we'll be using full matrix multiplies to implement the
        # unitary recurrence matrix
        Wparams_optim=Wparams
        IRe=np.eye(n_hidden).astype(np.float32)
        IIm=np.zeros((n_hidden,n_hidden)).astype(np.float32)
        Iaug=np.concatenate( [np.concatenate([IRe,IIm],axis=1),np.concatenate([IIm,IRe],axis=1)], axis=0)
        Waug=times_unitary(Iaug,n_hidden,swap_re_im,Wparams_optim,'adhoc')
        Wparams=[Waug]

    # extract recurrent parameters into this namespace 
    if flag_feed_forward:
        # just doing feed-foward, so remove any recurrent parameters
        if ('adhoc' in Wimpl):
            #theta = theano.shared(np.float32(0.0))
            h_0_size=(1,2*n_hidden)
            h_0 = theano.shared(np.asarray(np.zeros(h_0_size),dtype=theano.config.floatX))
        
        parameters = [V, U, hidden_bias, out_bias]
       
    else:
        if ('adhoc' in Wimpl):
            # restricted parameterization of Arjovsky, Shah, and Bengio 2015
            if ('fast' in Wimpl):
                theta = Wparams_optim[0]
                reflection = Wparams_optim[1]
                index_permute_long = Wparams_optim[2] 
            else:
                theta = Wparams[0]
                reflection = Wparams[1]
                index_permute_long = Wparams[2]

            parameters = [V, U, hidden_bias, reflection, out_bias, theta, h_0]
            #Wparams = [theta]
        elif (Wimpl == 'full'):
            # fixed full unitary matrix
            Waug=Wparams[0]

            parameters = [V, U, hidden_bias, out_bias, h_0, Waug]
            #Wparams = [Waug]

    h_0_all_layers = h_0

    # initialize additional layer parameters
    addl_layers_params=[]
    addl_layers_params_optim=[]
    for i_layer in range(2,n_layers+1):
        betw_layer_suffix='_L%d_to_L%d' % (i_layer-1,i_layer)
        layer_suffix='_L%d' % i_layer
        
        # create cross-layer unitary matrix
        Wvparams_cur = initialize_unitary(n_hidden,Wimpl,rng,name_suffix=(name_suffix+betw_layer_suffix),init=W_init)
        if (Wimpl=='adhoc_fast'):
            # create the full unitary matrix from the restricted parameters,
            # since we'll be using full matrix multiplies to implement the
            # unitary recurrence matrix
            Wvparams_cur_optim=Wvparams_cur
            IRe=np.eye(n).astype(np.float32)
            IIm=np.zeros((n_hidden,n_hidden)).astype(np.float32)
            Iaug=np.concatenate( [np.concatenate([IRe,IIm],axis=1),np.concatenate([IIm,IRe],axis=1)], axis=0)
            Wvaug=times_unitary(Iaug,n_hidden,swap_re_im,Wvparams_cur_optim,'adhoc')
            Wvparams_cur=[Wvaug]
        
        # create parameters for this layer
        hidden_bias_cur, h_0_cur, Wparams_cur = initialize_complex_RNN_layer(n_hidden,Wimpl,rng,hidden_bias_mean,name_suffix=(name_suffix + layer_suffix),hidden_bias_init=hidden_bias_init,h_0_init=h_0_init,W_init=W_init)
        if (Wimpl=='adhoc_fast'):
            # create the full unitary matrix from the restricted parameters,
            # since we'll be using full matrix multiplies to implement the
            # unitary recurrence matrix
            Wparams_cur_optim=Wparams_cur
            IRe=np.eye(n).astype(np.float32)
            IIm=np.zeros((n_hidden,n_hidden)).astype(np.float32)
            Iaug=np.concatenate( [np.concatenate([IRe,IIm],axis=1),np.concatenate([IIm,IRe],axis=1)], axis=0)
            Waug=times_unitary(Iaug,n_hidden,swap_re_im,Wparams_cur_optim,'adhoc')
            Wparams_cur=[Waug]
        
        addl_layers_params = addl_layers_params + Wvparams_cur + [hidden_bias_cur, h_0_cur] + Wparams_cur
        if (Wimpl=='adhoc'):
            # don't include permutation indices in the list of parameters to be optimized
            addl_layers_params_optim = addl_layers_params_optim + Wvparams_cur[0:2] + [hidden_bias_cur, h_0_cur] + Wparams_cur[0:2]
        elif (Wimpl=='adhoc_fast'):
            addl_layers_params_optim = addl_layers_params_optim + Wvparams_cur_optim[0:2] + [hidden_bias_cur, h_0_cur] + Wparams_cur_optim[0:2]
        else:
            addl_layers_params_optim = addl_layers_params

        h_0_all_layers = T.concatenate([h_0_all_layers,h_0_cur],axis=1)

    parameters = parameters + addl_layers_params_optim

    # initialize data nodes
    print "Every t: " , out_every_t
    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    print "My x: " , x
    if flag_use_mask:
        if 'CE' in loss_function:
            ymask = T.matrix(dtype='int8') if out_every_t else T.vector(dtype='int8')
        else:
            # y will be n_fram x n_output x n_utt
            ymask = T.tensor3(dtype='int8') if out_every_t else T.matrix(dtype='int8')

    if x_spec is not None:
        # x is specified, set x to this:
        x = x_spec
    


    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, ymask_t, h_prev, cost_prev, acc_prev, V, hidden_bias, out_bias, U, *argv):  
    	print "My x: " , x_t
        # h_prev is of size n_batch x n_layers*2*n_hidden

        # strip W parameters off variable arguments list
        if (Wimpl=='full') or (Wimpl=='adhoc_fast'):
            Wparams=argv[0:1]
            argv=argv[1:]
        else:
            Wparams=argv[0:3]
            argv=argv[3:]
        
        Wimpl_in_scan=Wimpl
        if (Wimpl=='adhoc_fast'):
            # just using a full matrix multiply is faster
            # than calling times_unitary with Wimpl='adhoc'
            Wimpl_in_scan='full'

        if not flag_feed_forward:
            # Compute hidden linear transform: W h_{t-1}
            h_prev_layer1 = h_prev[:,0:2*n_hidden]
            hidden_lin_output = times_unitary(h_prev_layer1,n_hidden,swap_re_im,Wparams,Wimpl_in_scan)

        # Compute data linear transform
        if ('CE' in loss_function) and (input_type=='categorical'):
            # inputs are categorical, so just use them as indices into V
            data_lin_output = V[T.cast(x_t, 'int32')]
        else:
            # second dimension of real-valued x_t should be of size n_input, first dimension of V should be of size n_input
            # (or augmented, where the dimension of summation is 2*n_input and V is of real/imag. augmented form)
            data_lin_output = T.dot(x_t, V)
            
        # Total linear output        
        if not flag_feed_forward:
            lin_output = hidden_lin_output + data_lin_output
        else:
            lin_output = data_lin_output

        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        #  add a little bit to sqrt argument to ensure stable gradients,
        #  since gradient of sqrt(x) is -0.5/sqrt(x)
        modulus = T.sqrt(1e-5+lin_output**2 + lin_output[:, swap_re_im]**2)
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
        h_t = lin_output * rescale
     
        h_t_all_layers = h_t

        # Compute additional recurrent layers
        for i_layer in range(2,n_layers+1):
            
            # strip Wv parameters off variable arguments list
            if (Wimpl=='full') or (Wimpl=='adhoc_fast'):
                Wvparams_cur=argv[0:1]
                argv=argv[1:]
            else:
                Wvparams_cur=argv[0:3]
                argv=argv[3:]
            
            # strip hidden_bias for this layer off argv
            hidden_bias_cur = argv[0]
            argv=argv[1:]
            
            # strip h_0 for this layer off argv
            #h_0_cur = argv[0] #unused, since h_0_all_layers is all layers' h_0s concatenated
            argv=argv[1:]
            
            # strip W parameters off variable arguments list
            if (Wimpl=='full') or (Wimpl=='adhoc_fast'):
                Wparams_cur=argv[0:1]
                argv=argv[1:]
            else:
                Wparams_cur=argv[0:3]
                argv=argv[3:]

            Wimpl_in_scan=Wimpl
            if (Wimpl=='adhoc_fast'):
                # just using a full matrix multiply is faster
                # than calling times_unitary with Wimpl='adhoc'
                Wimpl_in_scan='full'

            # Compute the linear parts of the layer ----------

            if not flag_feed_forward:
                # get previous hidden state h_{t-1} for this layer:
                h_prev_cur = h_prev[:,(i_layer-1)*2*n_hidden:i_layer*2*n_hidden]
                # Compute hidden linear transform: W h_{t-1}
                hidden_lin_output_cur = times_unitary(h_prev_cur,n_hidden,swap_re_im,Wparams_cur,Wimpl_in_scan)

            # Compute "data linear transform", which for this intermediate layer is the previous layer's h_t transformed by Wv
            data_lin_output_cur = times_unitary(h_t,n_hidden,swap_re_im,Wvparams_cur,Wimpl_in_scan)
                
            # Total linear output        
            if not flag_feed_forward:
                lin_output_cur = hidden_lin_output_cur + data_lin_output_cur
            else:
                lin_output_cur = data_lin_output_cur

            # Apply non-linearity ----------------------------

            # scale RELU nonlinearity
            #  add a little bit to sqrt argument to ensure stable gradients,
            #  since gradient of sqrt(x) is -0.5/sqrt(x)
            modulus = T.sqrt(1e-5+lin_output_cur**2 + lin_output_cur[:, swap_re_im]**2)
            rescale = T.maximum(modulus + T.tile(hidden_bias_cur, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
            h_t = lin_output_cur * rescale
            h_t_all_layers = T.concatenate([h_t_all_layers,h_t],axis=1)

        # assume we aren't passing any preactivation to compute_cost
        z_t = None

        if loss_function == 'MSEplusL1':
            z_t = h_t

        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
    
            if flag_add_input_to_output:
                lin_output=lin_output + x_t 
	    print "Lin output: " , lin_output.ndim
            if flag_use_mask:
                cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t, ymask_t=ymask_t, z_t=z_t, lam=lam)
            else:
                cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t, z_t=z_t, lam=lam)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t_all_layers, cost_t, acc_t
    
    # compute hidden states
    #  h_0_batch should be n_utt x n_layers*2*n_hidden, since scan goes over first dimension of x, which is the maximum STFT length in frames
    h_0_batch = T.tile(h_0_all_layers, [x.shape[1], 1])
    
    if input_type=='complex' and output_type=='complex':
        # pass in augmented input and output transformations
        non_sequences = [Vaug, hidden_bias, out_bias, Uaug] + Wparams + addl_layers_params
    elif input_type=='complex':
        non_sequences = [Vaug, hidden_bias, out_bias, Un] + Wparams + addl_layers_params
    elif output_type=='complex':
        non_sequences = [Vn   , hidden_bias, out_bias, Uaug] + Wparams + addl_layers_params
    else:
        non_sequences = [Vn   , hidden_bias, out_bias, Un] + Wparams + addl_layers_params
    
    if out_every_t:
        if flag_use_mask:
            sequences = [x, y, ymask]
        else:
            sequences = [x, y, T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    else:
        if flag_use_mask:
            sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1]), T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
        else:
            sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1]), T.tile(theano.shared(np.ones((1,1),dtype=theano.config.floatX)),[x.shape[0], 1, 1])]

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

    if flag_return_lin_output:
        if output_type=='complex':
            lin_output = T.dot(hidden_states, Uaug) + out_bias.dimshuffle('x',0)
        else:
            lin_output = T.dot(hidden_states, Un) + out_bias.dimshuffle('x',0)
   
        if flag_add_input_to_output:
            lin_output = lin_output + x
    #print "The lin output is: " , lin_output
    if not out_every_t:
        #TODO: here, if flag_use_mask is set, need to use a for-loop to select the desired time-step for each utterance
        lin_output = T.dot(hidden_states[-1,:,:], Un) + out_bias.dimshuffle('x', 0)
        z_t = None
        if loss_function == 'MSEplusL1':
            z_t = hidden_states[-1,:,:]
            print "Y:",y
        costs = compute_cost_t(lin_output, loss_function, y, z_t=z_t, lam=lam)
        cost=costs[0]
        accuracy=costs[1]
    else:
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
   
        if (loss_function=='CE_of_sum'):
            yest = T.sum(lin_output,axis=0) #sum over time_steps, yest is Nseq x n_output
            yest_softmax = T.nnet.softmax(yest)
            cost = T.nnet.categorical_crossentropy(yest_softmax, y[0,:]).mean()
            accuracy = T.eq(T.argmax(yest, axis=-1), y[0,:]).mean(dtype=theano.config.floatX)

    if flag_return_lin_output:

        costs = [cost, accuracy, lin_output]
        
        if flag_return_hidden_states:
            costs = costs + [hidden_states]

        #nmse_local = ymask.dimshuffle(0,1)*( (lin_output-y)**2 )/( 1e-5 + y**2 )
        nmse_local = theano.shared(np.float32(0.0))
        costs = costs + [nmse_local]

        costs = costs + [cost_steps]

    else:
        costs = [cost, accuracy]
    if flag_use_mask:
        return [x,y,ymask], parameters, costs
    else:
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




