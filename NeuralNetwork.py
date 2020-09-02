import numpy as np
import sys
from scipy.stats import truncnorm
import pickle as pkl
from datetime import datetime

def createPickleFile(trainImg, trainLabel, testImg):
    with open("train_image.pkl", "wb") as f:
        pkl.dump(trainImg,f)
        f.close()

    with open("train_label.pkl", "wb") as f:
        pkl.dump(trainLabel,f)
        f.close()

    with open("test_image.pkl", "wb") as f:
        pkl.dump(testImg,f)
        f.close()
    
def readPickleFile():
    f = open("train_image.pkl", "rb")
    trainImgs = pkl.load(f)
    f.close()

    f = open("train_label.pkl", "rb")
    trainLabels = pkl.load(f)
    f.close()

    f = open("test_image.pkl", "rb")
    testImgs = pkl.load(f)
    f.close()

    f = open("test_label.pkl", "rb")
    testLabels = pkl.load(f)
    f.close()

    return (trainImgs, trainLabels, testImgs, testLabels)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def crossEntropy(Y, A):
    return -np.mean(Y * np.log(A + 1e-8))

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

'''
    Write the predictions to csv
    params@
    testPredFile: filename
    predictions:  nparray or output
'''
def writeOutput(testPredFile, predictions):   
    np.savetxt(testPredFile, predictions, delimiter=',', fmt='%d')
    return True

'''
    Reads the input from the csv and convert to np array
'''
def readInput(trainImgFile, trainLabelFile, testImgFile):
    trainImg = normailize( np.loadtxt(trainImgFile, delimiter=','), 255)
    labels =  np.loadtxt(trainLabelFile, delimiter=',', dtype=int)
    trainLabel = oneHotEncoding(labels, 10)
    
    #test file
    testImg = normailize( np.loadtxt(testImgFile, delimiter=','), 255)
    return (trainImg, trainLabel, testImg)

'''
    the input array is converted to one hot format
    @params:
        input: array of integers
        numClass: number of diffrent classes
'''
def oneHotEncoding(input, numClass):
    return np.squeeze(np.eye(numClass)[input.reshape(-1)])

'''
    Nirmailze the data
    @params:
        input: array 
        max: max possible value in the array
'''
def normailize(input, max):
    return (input* 0.99/max) + 0.01

class NeuralNetwork:
    
    def __init__(self, L1, L2, L3, alpha, epochs, batch_size):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.epochs = epochs
        self.batch_size = 1
        self.alpha = alpha 
        self.init_weights()
        
    def init_weights(self):
        bound = 1 / np.sqrt(self.L1)
        X = truncated_normal(mean=0, sd=1, low=-bound, upp=bound)
        self.W1 = X.rvs((self.L2, self.L1))
                                       
        bound = 1 / np.sqrt(self.L2)
        X = truncated_normal(mean=0, sd=1, low=-bound, upp=bound)
        self.W2 = X.rvs((self.L3, self.L2))
    
    def train(self, X, Y):
        """
        X and Y can 
        be tuple, list or ndarray
        """
        
        X = np.array(X, ndmin=2).T
        Y = np.array(Y, ndmin=2).T
        a2, cache = self.feed_forward(X, Y)
        self.feed_backward( X, Y, cache)


    def SDG(self, trainImgs, trainLabels):
        for epoch in range(self.epochs):  
            print("epoch: ", epoch)
            for i in range(0, len(trainImgs), self.batch_size):
                self.train(trainImgs[i:(i+self.batch_size), :], trainLabels[i: (i+self.batch_size), : ])

            corrects, wrongs = self.evaluate(trainImgs, np.argmax(trainLabels, axis=1))
            print("accuracy train: ", corrects / ( corrects + wrongs))

       
    def feed_forward(self, X, Y):
        cache={}
        z1 = np.dot(self.W1, X)
        a1 = sigmoid(z1)
        cache['a1'] = a1
        z2 = np.dot(self.W2, a1)
        a2 = softmax(z2)
        cache['a2'] = a2
        L = crossEntropy(Y, a2)
        cache['L'] = L
        return a2, cache


    def feed_backward(self, X, Y, cache):
        a1 = cache['a1']
        a2 = cache['a2']
        L = a2 - Y
     
        delta = L * a2 * (1.0 - a2)     
        delta = self.alpha * np.dot(delta, a1.T)
        self.W2 -= delta
       
        delta_errors = np.dot(self.W2.T, L)
        delta = delta_errors * a1 * (1.0 - a1)
        delta = self.alpha * np.dot(delta, X.T)
        self.W1 -= delta
        
        
    
    def calculate(self, X):
        X = np.array(X, ndmin=2).T
        z1 = np.dot(self.W1, X)
        a1 = sigmoid(z1)
        z2 = np.dot(self.W2, a1)
        a2 = softmax(z2)
        return a2
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.calculate(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


    def predict(self, data):
        predictions = []
        for i in range(len(data)):
            res = self.calculate(data[i])
            predictions.append(res.argmax())
        
        return np.array(predictions)


if __name__ == "__main__":
    '''
        Define a Neural network, with params as:
        @params
        eta(learning rate), epoch, batch_size, inputD, trainImgFile, trainLabelFile, testLabFile, testPredFile
    '''
    start = datetime.now()

    # if len(sys.argv) < 3:
    #     print('Please provide file name')
    #     exit(1)
    
    # (trainImgs, trainLabels, testImgs) = readInput(sys.argv[1], sys.argv[2], sys.argv[3])
    (trainImgs, trainLabels, testImgs, testLabels) = readPickleFile()

    testPredFile = "test_predictions.csv"
    
    print(trainImgs.shape)
    print(trainLabels.shape)
    print(testImgs.shape)

    net = NeuralNetwork(L1 = 28*28, L2 = 100, L3 = 10, alpha = 0.1, epochs = 5,  batch_size=50)
    net.SDG(trainImgs, trainLabels)
    
    predictions = net.predict(testImgs)
    writeOutput("test_predictions.csv", predictions)
    
    print("============== Done Training ===================")
    print( (datetime.now() - start).total_seconds()) 
    corrects, wrongs = net.evaluate(testImgs, np.argmax(testLabels, axis=1))
    print("accuracy train: ", corrects / ( corrects + wrongs))
