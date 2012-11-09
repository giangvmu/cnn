from numpy import array
from numpy import loadtxt
import RNNLib
import CNNLib
import MCNNLib

def RNN_run(ni,nh,no,eps,lr,mm,f_train,f_test):
    num_input=ni
    num_hidden=nh
    num_output=no
    epochs=eps
    l_rate=lr
    moment=mm
    file_train=f_train
    file_test=f_test

    # read data from data to train
    data = loadtxt( file_train, delimiter = ' ')
    inp_data =  data[:, :num_input] #first num_input columns
    out_data = data[:, num_input:] #last num_output column
    inp_train=[]
    for i in range(len(inp_data)):
        inp_train.append([list(inp_data[i]),list(out_data[i])])

    # create a network with two input, two hidden, and one output nodes
    RNet = RNNLib.NN(num_input, num_hidden, num_output, regression = False)

    # train it with some patterns
    tmp_err=RNet.train(inp_train, epochs, l_rate, moment, True)
     # read data from data to test
    data = loadtxt( file_test, delimiter = ' ')
    inp_data =  data[:, :num_input] #first num_input columns
    out_data = data[:, num_input:] #last num_output column
    inp_test=[]
    for i in range(len(inp_data)):
        inp_test.append([list(inp_data[i])])

    results = RNet.test(inp_test, verbose = False)
    results=array(results)
    #RNNLib.plot(out_data,results,num_output)
    return tmp_err

def CNN_run(ni,nh,no,eps,lr,mm,f_train,f_test):
    num_input=ni
    num_hidden=nh
    num_output=no
    epochs=eps
    l_rate=lr
    moment=mm
    file_train=f_train
    file_test=f_test
    # read data from data to train
    data = loadtxt(file_train, delimiter = ' ')
    inp_data =  data[:, :num_input*2] #first num_input columns
    out_data = data[:, num_input*2:] #last num_output column
	
    #Convert input float to complex
    cinp_data=[]
    for i in range(len(inp_data)):
         ctmp=[]	
         for j in range(num_input):
            ctmp.append(complex(inp_data[i,j*2],inp_data[i,j*2+1]))
         cinp_data.append(ctmp)		

    #Convert output float to complex		 
    cout_data=[] 		 
    for i in range(len(out_data)):
         ctmp=[]	
         for j in range(num_output):
            ctmp.append(complex(out_data[i,j*2],out_data[i,j*2+1]))
         cout_data.append(ctmp)	

    #Convert input, output to train 
    inp_train=[]	 
    for i in range(len(cinp_data)):
         inp_train.append([list(cinp_data[i]),list(cout_data[i])])
    #print inp_train

    # create a network with two input, two hidden, and one output nodes
    CNet = CNNLib.NN(num_input, num_hidden, num_output, regression = False)

    # train it with some patterns
    tmp_err=CNet.train(inp_train, epochs, l_rate, moment, True)

	# read data from data to test
    data = loadtxt( file_test, delimiter = ' ')
    inp_data =  data[:, :num_input*2] #first num_input columns
    out_data = data[:, num_input*2:] #last num_output column

    #Convert input float to complex
    cinp_data=[]
    for i in range(len(inp_data)):
         ctmp=[]	
         for j in range(num_input):
            ctmp.append(complex(inp_data[i,j*2],inp_data[i,j*2+1]))
         cinp_data.append(ctmp)		

    #Convert input to test
    inp_test=[]
    for i in range(len(cinp_data)):
        inp_test.append([list(cinp_data[i])])

	results = CNet.test(inp_test, verbose = False)
    results=array(results)
    tmp_out=[]
	
    #Convert results-complex to array-real	
    for i in range(len(results)):
        tmp=[]	
        for j in range(num_output):
            tmp.append(results[i,j].real)		
            tmp.append(results[i,j].imag) 			
        tmp_out.append(tmp)
    tmp_out=array(tmp_out)
    #CNNLib.plot(out_data,tmp_out,num_output)
    return tmp_err

def MCNN_run(ni,nh,no,eps,lr,mm,f_train,f_test):
    num_input=ni
    num_hidden=nh
    num_output=no
    epochs=eps
    l_rate=lr
    moment=mm
    file_train=f_train
    file_test=f_test
    # read data from data to train
    data = loadtxt(file_train, delimiter = ' ')
    inp_data =  data[:, :num_input*2] #first num_input columns
    out_data = data[:, num_input*2:] #last num_output column

    #Convert input float to complex
    cinp_data=[]
    for i in range(len(inp_data)):
         ctmp=[]
         for j in range(num_input):
            ctmp.append(complex(inp_data[i,j*2],inp_data[i,j*2+1]))
         cinp_data.append(ctmp)

    #Convert output float to complex
    cout_data=[]
    for i in range(len(out_data)):
         ctmp=[]
         for j in range(num_output):
            ctmp.append(complex(out_data[i,j*2],out_data[i,j*2+1]))
         cout_data.append(ctmp)

    #Convert input, output to train
    inp_train=[]
    for i in range(len(cinp_data)):
         inp_train.append([list(cinp_data[i]),list(cout_data[i])])
    #print inp_train

    # create a network with two input, two hidden, and one output nodes
    MCNN_Net = MCNNLib.MNN(num_input, num_hidden, num_output, regression = False)

    # train it with some patterns
    tmp_err=MCNN_Net.train(inp_train, epochs, l_rate, moment, True)

	# read data from data to test
    data = loadtxt( file_test, delimiter = ' ')
    inp_data =  data[:, :num_input*2] #first num_input columns
    out_data = data[:, num_input*2:] #last num_output column

    #Convert input float to complex
    cinp_data=[]
    for i in range(len(inp_data)):
         ctmp=[]
         for j in range(num_input):
            ctmp.append(complex(inp_data[i,j*2],inp_data[i,j*2+1]))
         cinp_data.append(ctmp)

    #Convert input to test
    inp_test=[]
    for i in range(len(cinp_data)):
        inp_test.append([list(cinp_data[i])])

	results = MCNN_Net.test(inp_test, verbose = False)
    results=array(results)
    tmp_out=[]

    #Convert results-complex to array-real
    for i in range(len(results)):
        tmp=[]
        for j in range(num_output):
            tmp.append(results[i,j].real)
            tmp.append(results[i,j].imag)
        tmp_out.append(tmp)
    tmp_out=array(tmp_out)
    #MCNNLib.plot(out_data,tmp_out,num_output)
    return tmp_err


if __name__ == '__main__':
    err1=RNN_run(4,5,4,300,0.2,0.1,'data/train3.txt','data/test3.txt')
    err2=CNN_run(2,5,2,300,0.2,0.1,'data/train3.txt','data/test3.txt')
    err3=MCNN_run(2,5,2,300,0.2,0.1,'data/train3.txt','data/test3.txt')

    #Plot error
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize = (10,8))
        plt.subplots_adjust(hspace=0.4)
        plt.plot( err1[0:], 'b-' )
        plt.plot( err2[0:], 'r-' )
        plt.plot( err3[0:], 'k-' )
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.grid(True)
        plt.legend(('Real NN', 'Complex NN', 'Modify complex NN'))
        plt.show()
    except ImportError, e:
        print "Cannot make plots. For plotting install matplotlib.\n%s" % e