#This file exports a saved file from my version into the version for Li.
#The only thing is I still need to figure out the parameter count.
#We are going to just put ~6500, cause screw it.
import cPickle
import numpy as np

'''
First run

filename = "John_Bengio_T=1000_6500.txt"
histfile = "memory_problem_final_complex_RNN_adhoc_complex_RNN_nhidden128_t1000"
title="Bengio URNN with ~6500 parameters on"
'''
'''
filename = "Full_Assoc_T=1000_6500.txt"
histfile = 'memory_problem_final_complex_RNN_full_complex_RNN_nhidden65_t1000'
title="Fully Associative URNN with ~6500 parameters on"
task="# Copying Task with T=1000"
'''
filename = "John_LSTM_T=1000_6500.txt"
histfile = 'memory_problem_final_LSTM_adhoc_LSTM_nhidden40_t1000'
title="LSTM with ~6500 parameters on"
task="# Copying Task with T=1000"


beginning_string='########\n\n\tModel: '+str(title)+'\n\n#Task:\n '+str(task)+'\n\n########'

try:
    history=cPickle.load(open(histfile,'rb'))
except:
    print ("Can't open file",histfile,"skipping exp:",filename)
train_loss=np.asarray(history['train_loss'])
xval=np.arange(train_loss.shape[0])
xtrain = np.array(range(0,len(train_loss)))*128.0
#x is xtrain, y is train_loss

with open(filename,'w') as f:
	f.write(beginning_string)
	f.write('\n\n');
	for i in xrange(0,len(train_loss)):
		if str(train_loss[i]) == 'nan':
			continue
		else:
			f.write(str(xtrain[i])+"	" + str(train_loss[i]) + "\n")