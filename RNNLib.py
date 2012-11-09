import math
import random
import cPickle
from numpy import array
from numpy import loadtxt
import zipfile,os

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    #if x>50: return 1
    #if x<-50: return 0
    #output = 1.0/(1+math.exp(-x))
    #return output
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
	#return y*(1 - y)
    return 1.0 - y**2

def esigmoid(y):
    return 1.0/(1- math.exp(-y))

def plot(outputs_real, outputs_nn,num_output):
    #Plot a given function.
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize = (10,8))
        plt.subplots_adjust(hspace=0.4)
        for i in range(num_output):
            plt.subplot(num_output,1,i)
            plt.plot( outputs_real[:,i:i+1], 'b--' )
            plt.plot( outputs_nn[:,i:i+1], 'k-' )
            #plt.xlabel("X values")
            plt.ylabel("Target, output")
            plt.title("Value "+str(i+1))
            plt.grid(True)
            plt.legend(('Target', 'Output'))
        plt.show()
    except ImportError, e:
        print "Cannot make plots. For plotting install matplotlib.\n%s" % e
    

class NN:
    def __init__(self, ni, nh, no, regression = False):
        """NN constructor.
        ni, nh, no are the number of input, hidden and output nodes.
        regression is used to determine if the Neural network will be trained 
        and used as a classifier or for function regression.
        """
        
        self.regression = regression
        
        #Number of input, hidden and output nodes.
        self.ni = ni  + 1 # +1 for bias node
        self.nh = nh  + 1 # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-1, 1)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-1, 1)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)


    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError, 'wrong number of inputs'

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh - 1):
            total = 0.0
            for i in range(self.ni):
                total += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(total)

        # output activations
        for k in range(self.no):
            total = 0.0
            for j in range(self.nh):
                total += self.ah[j] * self.wo[j][k]
            self.ao[k] = total
            if not self.regression:
                self.ao[k] = sigmoid(total)
        
        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError, 'wrong number of target values'

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            output_deltas[k] = targets[k] - self.ao[k]
            if not self.regression:
                output_deltas[k] = dsigmoid(self.ao[k]) * output_deltas[k]

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += ((targets[k]-self.ao[k])**2)
        return 0.5*error


    def test(self, patterns, verbose = False):
        tmp = []
        for p in patterns:
            if verbose:
                print p[0], '->', self.update(p[0])
            tmp.append(self.update(p[0]))
        return tmp

    def weights_print(self):
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]
        print
        
    def weights_copy(self):
        tmp=[]
        for i in range(self.ni):
            for j in range(self.nh):
                tmp.append(self.wi[i][j])
        for j in range(self.nh):
            for k in range(self.no):
                tmp.append(self.wo[j][k])
        return tmp

    def weights_update(self,new_w):
        for i in range(self.ni):
            for j in range(self.nh):
               self.wi[i][j]=new_w[i*self.nh+j]
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k]=new_w[self.ni*self.nh+j*self.no+k]


    def train(self, patterns, iterations=1, N=0.5, M=0.1, verbose = False):
        """Train the neural network.
        N is the learning rate.
        M is the momentum factor.
        """
        tmp_err=[]
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                self.update(p[0])
                tmp = self.backPropagate(p[1], N, M)
                error += tmp
            if ((i+1) % 100 == 0) and (verbose==True):
                print '%s - error:%s ' %(i+1,error)	
            tmp_err.append(error)
        return tmp_err
    
    def P_Error(self,target):
        x_error = float(0.0)
        for k in range(len(target)):
            x_error = x_error + 0.5 * ((target[k]-self.ao[k])**2)
        return x_error

    def GetError(self,patterns):
        x_error = float(0.0)
        for p in patterns:
            inputs  = p[0]
            targets = p[1]
            self.update(inputs)
            x_error = x_error + self.P_Error(targets)
            #x_error = x_error + self.backPropagate(targets, 0.5, 0.1)
        return x_error

    def SaveW(self,filename):
         W = [self.wi,self.wo]
         cPickle.dump(W,open(filename,'w'))


    def LoadW(self,filename):
         W = cPickle.load(open(filename,'r'))
         self.wi=W[0]
         self.wo=W[1]
    
def savenet(net, filename):
    #Dumps network to a file using cPickle.
    import cPickle
    file = open(filename, 'w')
    cPickle.dump(net, file)
    file.close()
    return

def loadnet(filename):
    #Loads network pickled previously with `savenet`.
    import cPickle
    file = open(filename, 'r')
    net = cPickle.load(file)
    return net

#New - functions for program on server and client
def extract(zipfilepath, extractiondir):
    zip = zipfile.ZipFile(zipfilepath)
    zip.extractall(path=extractiondir)

def zip(textfile_in,zipfile_out):
    # save the files in .zip file
    zout = zipfile.ZipFile(zipfile_out, "w",compression=zipfile.ZIP_DEFLATED,)
    zout.write(textfile_in)
    zout.close()

def init_net(file_net,num_input,num_hidden,num_output):

    #Set patterns, input, hidden, output (From tasks) to create net
    net = NN(num_input, num_hidden, num_output, regression = True)
    #Save net
    savenet(net,file_net)

def store_weights(file_net,file_weights):
    net=loadnet(file_net)
    old_w=net.weights.copy()
    #Store weights
    f=open(file_weights,'w')
    for k in range(len(ole_w)):
        f.write('%s '%(old_w[k]))
    f.close()
    