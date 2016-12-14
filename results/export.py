#This file exports a saved file from my version into the version for Li.
#The only thing is I still need to figure out the parameter count.
#We are going to just put ~6500, cause screw it.
import cPickle
import numpy as np

T = 20
batch_size=128
direc='Copy_T=20_Full_URNN_LSTM_6500params/'


filename = "John_Bengio_T="+str(T)+"_6500.txt"
histfile = direc+"memory_problem_complex_RNN_adhoc_complex_RNN_nhidden128_t"+str(T)
title="Bengio URNN with ~6500 parameters on"
task="# Copying Task with T="+str(T)

'''
filename = "Full_Assoc_T="+str(T)+"_6500.txt"
histfile = direc+'memory_problem_complex_RNN_full_complex_RNN_nhidden64_t'+str(T)
title="Fully Associative URNN with ~6500 parameters on"
task="# Copying Task with T="+str(T)
'''

'''
filename = "John_LSTM_T="+str(T)+"_6500.txt"
histfile = direc+'memory_problem_LSTM_adhoc_LSTM_nhidden40_t'+str(T)
title="LSTM with ~6500 parameters on"
task="# Copying Task with T="+str(T)
'''


beginning_string='########\n\n\tModel: '+str(title)+'\n\n#Task:\n '+str(task)+'\n\n########'

try:
    history=cPickle.load(open(histfile,'rb'))
except:
    print ("Can't open file",histfile,"skipping exp:",filename)
train_loss=np.asarray(history['train_loss'])
xval=np.arange(train_loss.shape[0])
xtrain = np.array(range(0,len(train_loss)))*batch_size
#x is xtrain, y is train_loss

with open(filename,'w') as f:
	f.write(beginning_string)
	f.write('\n\n');
	for i in xrange(0,len(train_loss)):
		if str(train_loss[i]) == 'nan':
			continue
		else:
			f.write(str(xtrain[i])+"	" + str(train_loss[i]) + "\n")