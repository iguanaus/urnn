import theano, cPickle
import theano.tensor as T
import numpy as np
from fftconv import cufft, cuifft


#This initializes a n_in by n_out matrix and sets its name
def initialize_matrix(n_in,n_out,name,init='rand'):
	if (init=='rand'):
		sigma = np.sqrt(6. / (n_in+n_out))
		values = np.asarray(np.random.uniform(low=-sigma,high=sigma,size=(n_in,n_out)),dtype=theano.config.floatX)
	return theano.shared(value = values, name=name)

#This creates a unitary matrix based on Bengio's paper. This is just an initialize thingy.
#I assume that this is adhoc
def initialize_unitary(dim_size,name ="H"):
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



def initialize_complex_RNN_layer(n_hidden,hidden_bias_mean=0,name="H"):
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

#Just dot product them for now (this is incredibly slow)
def times_unitary(x,n,swap_re_im,Wparams):
	return T.dot(x,Wparams[0])

#This is averaged?!?!??!?!?!?
def compute_cost_t(lin_output,loss_function,y_t):
	RNN_output = T.nnet.softmax(lin_output)
	CE = T.nnet.categorical_crossentropy(RNN_output,y_t)
	cost_t = CE.mean()
	acc_t = (T.eq(T.argmax(RNN_output, axis=-1),y_t)).mean(dtype=theano.config.floatX)
	return cost_t, acc_t




if __name__=="__main__":
	initialize_unitary(10,"Hidden")

