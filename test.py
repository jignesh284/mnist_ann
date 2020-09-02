import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import truncnorm
import pickle as pkl
from datetime import datetime
 
class ANN:
    def __init__(self, layers_size, alpha=0.01 ,epochs=5):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.alpha = alpha
        self.epochs = epochs
        self.costs = []
 
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
 
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
    
   
    def initialize_parameters(self):
        for l in range(1, len(self.layers_size)):
            bound = 1.0/ np.sqrt(self.layers_size[l - 1])
            X = truncated_normal(mean=0, sd=1, low=-bound, upp=bound)
            self.parameters["W" + str(l)] = X.rvs((self.layers_size[l], self.layers_size[l - 1])) 
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
 
    def forward(self, X):
        store = {}
        A = X.T
        for l in range(self.L - 1):
            Z = np.dot(self.parameters["W" + str(l + 1)],  A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z
 
        Z = np.dot(self.parameters["W" + str(self.L)], A) + self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z
 
        return A, store
 
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)
 
    def backward(self, X, Y, store):
 
        derivatives = {}
        store["A0"] = X.T
        A = store["A" + str(self.L)]
        dZ = A - Y.T
 
        dW = np.dot(dZ, store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = np.dot(store["W" + str(self.L)].T, dZ)
 
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
 
        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / self.n * np.dot(dZ ,store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = np.dot(store["W" + str(l)].T, dZ)
 
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
 
        return derivatives
 
    def fit(self, X, Y):
        self.n = X.shape[0]
        self.layers_size.insert(0, X.shape[1])
        self.initialize_parameters()

        for epoch in range(self.epochs):
            #shuffle's the array
            shuffle(X, Y)
            cost=0
            batch=50
            start=0 
            end = start+batch
            N = len(X)
            while end < N:
                NEW_X = X[start:end, :]
                NEW_Y = Y[start:end, :]
                cost = self.run(NEW_X, NEW_Y)
                start = end
                end = start+batch 
 
            
            print("E : ",epoch , " Cost: ", cost, "Train Accuracy:", self.accuracy(Y, self.predict(X, Y)) )
            self.costs.append(cost)

    def run(self, X, Y):
        A, store = self.forward(X)
        cost = -np.mean(Y * np.log(A.T+ 1e-8))
        derivatives = self.backward(X, Y, store)
 
        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.alpha * derivatives["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.alpha * derivatives["db" + str(l)]

        return cost

 
    def predict(self, X, Y):
        A, cache = self.forward(X)
        Y_hat = np.argmax(A, axis=0)
        return Y_hat

    def accuracy(self, Y, Y_hat):
        Y = np.argmax(Y, axis=1)
        accuracy = (Y_hat == Y).mean()
        return accuracy * 100
 
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()
 
 


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

'''
    Reads the input from the csv and convert to np array
'''
def readInput(trainImgFile, trainLabelFile, testImgFile, testLabelFile):
    trainImg = normailize( np.loadtxt(trainImgFile, delimiter=','), 255)
    labels =  np.loadtxt(trainLabelFile, delimiter=',', dtype=int)
    trainLabel = oneHotEncoding(labels, 10)

    #test file
    testImg = normailize( np.loadtxt(testImgFile, delimiter=','), 255)

    test_labels =  np.loadtxt(testLabelFile, delimiter=',', dtype=int)
    testLabel = oneHotEncoding(test_labels, 10)

    return (trainImg, trainLabel, testImg, testLabel)

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

def shuffle(X, Y):
    np.random.seed(0)
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

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
 
 
if __name__ == '__main__':

    # train_x, train_y, test_x, test_y = readInput("train_image.csv", "train_label.csv", "test_image.csv", "test_label.csv")
    start = datetime.now()

    train_x, train_y, test_x, test_y  = readPickleFile()
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    print("train_y's shape: " + str(train_y.shape))
    print("test_y's shape: " + str(test_y.shape))
 
    layers_dims = [100, 10]
 
    ann = ANN(layers_dims, alpha=0.1, epochs=200)
    ann.fit(train_x, train_y)
    print("Train Accuracy:", ann.accuracy( train_y, ann.predict(train_x, train_y)))
    print("Test Accuracy:", ann.accuracy( test_y, ann.predict(test_x, test_y)) )
    # ann.plot_cost()

    print("============== Done Training ===================")
    print( (datetime.now() - start).total_seconds()) 