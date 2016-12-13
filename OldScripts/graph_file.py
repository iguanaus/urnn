#This takes in a datafile and graphs it
import matplotlib.pyplot as plt
input_file = "exp/LSTM_param_7967.txt"

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

plt.plot(xvals,yvals)
plt.show()







    # w, h = [int(x) for x in next(f).split()] # read first line
    # array = []
    # for line in f: # read rest of lines
    #     array.append([int(x) for x in line.split()])

