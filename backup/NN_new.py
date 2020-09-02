import numpy as np
import matplotlib.pyplot as plt

arch = [
  {"size": 784, "activation": "none"}, # input layer
  {"size": 16, "activation": "sigmoid"},
  {"size": 16, "activation": "sigmoid"},
  {"size": 10, "activation": "softmax"},
 ]

def init(arch, seed=3):
    parameters = {}
    number_of_layers = len(arch)

    for i in range(1, number_of_layers):
        parameters['W'+str(i)] = np.random.randn( (arch[i]["size"], arch[i-1]["size"]) ) * 0.01
        parameters['b'+str(i)] = np.zeroes( (arch[i]["size"], 1) )

    return parameters

def sigmoid(Z):
    val = 1 / (1+np.exp(-Z))
    return val

def delta_sigmoid(dA, s):
    s = sigmoid(s)
    ds = s*(1 - s)
    return dA * ds

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

# def delta_softmax(s):
#     for i in range(len(self.value)):
#         for j in range(len(self.input)):
#             if i == j:
#                 self.gradient[i] = self.value[i] * (1-self.input[i))
#             else: 
#                  self.gradient[i] = -self.value[i]*self.input[j]

def model_backward(X, parameters, arch):
    forward_cache = {}
    A=X
    number_of_layers = len(arch)

    for i in range(1, number_of_layers ):
        A_prev = A
        W = parameters['W'+str(i)]
        b = parameters['b'+str(i)]
        activation = arch[i]["activation"]
        Z, A = linear_activation_forward(A_prev, W, b, activation)
        forward_cache['Z' + str(l)] = Z
        forward_cache['A' + str(l-1)] = A
    
    AL=A

    return AL, forward_cache    


def linear_activation_forward(A_prev, W, b, activation):
    Z = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "softmax":
        A = softmax(Z)

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b 
    return Z

def compute_cost(AL, Y):
    m = Y.shape[1]
    # Compute loss from AL and y
    logprobs = np.multiply(np.log(AL),Y) + np.multiply(1 - Y, np.log(1 - AL))
    # cross-entropy cost
    cost = - np.sum(logprobs) / m
    cost = np.squeeze(cost)

    return cost


def L_model_backward(AL, Y, parameters, forward_cache, nn_architecture):
    grads = {}
    number_of_layers = len(nn_architecture)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dA_prev = dAL

    for l in reversed(range(1, number_of_layers)):
        dA_curr = dA_prev

        activation = nn_architecture[l]["activation"]
        W_curr = parameters['W' + str(l)]
        Z_curr = forward_cache['Z' + str(l)]
        A_prev = forward_cache['A' + str(l-1)]

        dA_prev, dW_curr, db_curr = linear_activation_backward(dA_curr, Z_curr, A_prev, W_curr, activation)

        grads["dW" + str(l)] = dW_curr
        grads["db" + str(l)] = db_curr

    return grads

def linear_activation_backward(dA, Z, A_prev, W, activation):
    if activation == "softmax":
        dZ = delta_softmax(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, A_prev, W)
    elif activation == "sigmoid":
        dZ = delta_sigmoid(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, A_prev, W)

    return dA_prev, dW, db

def linear_backward(dZ, A_prev, W):
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def model( X, Y, arch, alpha=0.01, epoch=10):
    np.random.seed(1)
    costs=[]

    parameters = init(arch)


    for i in range(0, epoch):

        AL, forward_cache = model_forward(X, parameters, arch)

        cost = compute_cost(AL, Y)

        grads = model_backward(AL, Y, parameters, forward_cache, arch)

        parameters = update_paramteres(parameters, grads, alpha)


