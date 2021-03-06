import math
import random
from numpy import array
from numpy import loadtxt

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([complex(fill,fill)]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)+j*1/(1+e^-y)
def sigmoid1(c):
    output=complex(math.tanh(c.real),math.tanh(c.imag))   
    return output

def sigmoid(c):
    output=complex(1.0/(1+math.exp(-c.real)),1.0/(1+math.exp(-c.imag)))   
    return output

def sigmoid2(c):
    output=complex(math.tanh(c.real)/(1-(c.real-3)*math.exp(-c.real)),math.tanh(c.imag)/(1-(c.imag-3)*math.exp(-c.imag)))   
    return output
	
# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(c):	
    return (1+sigmoid(c))*sigmoid(c)

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
        self.ai = [complex(1.0,1.0)]*self.ni
        self.ah = [complex(1.0,1.0)]*self.nh
        self.ao = [complex(1.0,1.0)]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = complex(rand(-1, 1),rand(-1, 1))
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = complex(rand(-1, 1),rand(-1, 1))

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
            total = complex(0.0,0.0)
            for i in range(self.ni):
                total += self.ai[i] * self.wi[i][j]  
            #print self.wi
            self.ah[j] = sigmoid(total)

        # output activations
        for k in range(self.no):
            total = complex(0.0,0.0)
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
        output_deltas = [complex(0.0,0.0)] * self.no
        for k in range(self.no):
            output_deltas[k] = targets[k] - self.ao[k]
            if not self.regression:
                output_deltas[k] = complex(output_deltas[k].real*(1-self.ao[k].real)*self.ao[k].real,output_deltas[k].imag*(1-self.ao[k].imag)*self.ao[k].imag)				
				
        # calculate error terms for hidden
        hidden_deltas = [complex(0.0,0.0)] * self.nh
        for j in range(self.nh):
            tmpreal=0.0			
            tmpimag=0.0			
            for k in range(self.no):
                tmpreal+=output_deltas[k].real*(1-self.ao[k].real)*self.ao[k].real*self.wo[j][k].real+output_deltas[k].imag*(1-self.ao[k].imag)*self.ao[k].imag*self.wo[j][k].imag
                tmpimag+=output_deltas[k].real*(1-self.ao[k].real)*self.ao[k].real*self.wo[j][k].imag-output_deltas[k].imag*(1-self.ao[k].imag)*self.ao[k].imag*self.wo[j][k].real				
            hidden_deltas[j] = complex((1-self.ah[j].real) *self.ah[j].real* tmpreal,-1.0*(1-self.ah[j].imag) *self.ah[j].imag* tmpimag)				

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                #change = self.ah[j]*output_deltas[k]
                change = math.sqrt(self.ah[j].real*self.ah[j].real+self.ah[j].imag*self.ah[j].imag)*output_deltas[k]				
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                #change = hidden_deltas[j]*self.ai[i]
                change = math.sqrt(self.ai[i].real*self.ai[i].real+self.ai[i].imag*self.ai[i].imag)*hidden_deltas[j]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        #error = complex(0.0,0.0)
        error = 0.0		
        for k in range(len(targets)):
            #error += ((targets[k]-self.ao[k])*(targets[k]-self.ao[k]))
            error += ((targets[k].real-self.ao[k].real)*(targets[k].real-self.ao[k].real)+(targets[k].imag-self.ao[k].imag)*(targets[k].imag-self.ao[k].imag))
        return 0.5*error


    def test(self, patterns, verbose = False):
        tmp = []
        for p in patterns:
            if verbose:
                print p[0], '->', self.update(p[0])
            tmp.append(self.update(p[0]))
        return tmp

    def train(self, patterns, iterations=1, N=0.5, M=0.1, verbose = False):
        """Train the neural network.
        N is the learning rate.
        M is the momentum factor.
        """
        tmp_error=[]
        for i in xrange(iterations):
            #error = complex(0.0,0.0)
            error = 0.0
            for p in patterns:
                self.update(p[0])
                tmp = self.backPropagate(p[1], N, M)
                error += tmp
            if ((i+1) % 100 == 0) and (verbose==True):
                #print '%s - error real:%s , imaginary: %s' %(i+1,error.real,error.imag)
                print '%s - error:%s ' %(i+1,error)				
            tmp_error.append(error)
        return tmp_error

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

def plot(outputs_real, outputs_nn,num_output):
    #Plot a given function.
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize = (10,8))
        plt.subplots_adjust(hspace=0.4)
        for i in range(num_output):
            plt.subplot(2,1,i)
            plt.plot( outputs_real[:,i:i+2], 'b--' )
            plt.plot( outputs_nn[:,i:i+2], 'k-' )
            plt.xlabel("X values")
            plt.ylabel("Target, output")
            plt.title("Value "+str(i+1))
            plt.grid(True)
            plt.legend(('Target', 'Output'))
        plt.show()
    except ImportError, e:
        print "Cannot make plots. For plotting install matplotlib.\n%s" % e
