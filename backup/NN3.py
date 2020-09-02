import sys
import numpy as np
import scipy as sk
import matplotlib as plt
from datetime import datetime

# start = datetime.now()
# writeOutput("test_predictions.csv", np.zeros((1000,), dtype=int))
# print( (datetime.now() - start).total_seconds() )


class NeuralNetwork:

    def __init__(self, eta, epochs, batch_size, inputD, trainImgFile, trainLabelFile, testImgFile, testPredFile):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.dimes = [inputD]
        self.layers = []
        self.testPredFile = testPredFile
        (self.trainImg, self.trainLabel, self.testImg) = readInput(trainImgFile, trainLabelFile, testImgFile)
    
    def init_params(self):
        params = {"W1": np.random.randn(opt.n_h, opt.n_x) * np.sqrt(1. / opt.n_x),
          "b1": np.zeros((opt.n_h, 1)) * np.sqrt(1. / opt.n_x),
          "W2": np.random.randn(digits, opt.n_h) * np.sqrt(1. / opt.n_h),
          "b2": np.zeros((digits, 1)) * np.sqrt(1. / opt.n_h)}

        return params

    def addLayer(self, layer):
        self.layers.append(layer)

    def train(self):
        print()
        
    def predict(self):
        print()

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def forward_prop(self, x):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x)+b)
        return x

    def backward_prop(self):
        print("backward")
        

    def accuracy(self, predictions, actual):
        error = np.count_nonzero((predictions - actual))
        return 1 - (error/len(actual)); 
    
    def SGD(self, training_data, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            for k in range(0, n, self.batch_size):
                if (k+self.batch_size) <= n:
                    batches =  training_data[ k : , :]
                    for batch in batches:
                        self.update_mini_batch(batch, self.eta)
                    if test_data:
                        print("Epoch {0}: {1} / {2}".format(
                            j, self.evaluate(test_data), n_test))
                    else:
                        print("Epoch {0} complete".format(j))

    


class Layer:

    def __init__(self, n):
        self.n = n
        self.w = np.zeros((self.n,))
        self.b = np.zeros((self.n,))

'''
    individual funtion
'''
def crossEntropy(act, pred):
    avg = np.sum(np.multiply( act, np.log(pred)))
    m = act.shape[1]
    L = -(1./m) * avg
    return L

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def delta_sigmoid(z):
    """Derivative of the sigmoid function."""
    val = sigmoid(z)
    return val*(1 - val)

# def softmax(s):
#     exps = np.exp(s - np.max(s, axis=1, keepdims=True))
#     return exps/np.sum(exps, axis=1, keepdims=True)

'''
    Write the predictions to csv
    params@
    testPredFile: filename
    predictions:  nparray or output
'''
def writeOutput(testPredFile, predictions):   
    #predictions = np.argmax(predictions, axis=1)
    np.savetxt(testPredFile, predictions, delimiter=',', fmt='%d')
    return True

'''
    Reads the input from the csv and convert to np array
'''
def readInput(trainImgFile, trainLabelFile, testImgFile):
    trainImg = normailize( np.loadtxt(trainImgFile, delimiter=','), 255)
    labels =  np.loadtxt(trainLabelFile, delimiter=',', dtype=int)
    trainLabel = np.squeeze(np.eye(10)[labels.reshape(-1)])
    #test file
    testImg = normailize( np.loadtxt(testImgFile, delimiter=','), 255)
    return (trainImg, trainLabel, testImg)



'''
    @params
        inp: values to normalize
        mVal: maximum possible input value
    @return
        value ranging from 0 to 1
'''
def normailize(inp, mVal):
    return inp/mVal






if __name__ == "__main__":
    '''
        Define a Neural network, with params as:
        @params
        eta(learning rate), epoch, batch_size, inputD, trainImgFile, trainLabelFile, testLabFile, testPredFile
    '''
    if len(sys.argv) < 4:
        print('Please provide file name')
        exit(1)
 
        
    nn = NeuralNetwork( 0.1, 5, 32, 784, sys.argv[1], sys.argv[2], sys.argv[3], "test_predictions.csv")
