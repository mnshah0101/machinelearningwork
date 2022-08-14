#deep neural net which differentiates between cat and dog pictures

import pickle #import for pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
#read pickle files
X = pickle.load(open("X.pickle", 'rb'))
y = pickle.load(open("y.pickle", 'rb'))

X= np.array(X).reshape(-1, 100, 100, 1) #reshape so it becomes 25000 images that are 100x100x1
#featuring, 255 is the highest value
X = X/255
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.95,random_state=101)
X_test = X_test.reshape(10000,23699)
y_test = y_test.reshape(1,23699)
X_train = X_train.reshape(10000,1247)
y_train = y_train.reshape(1,1247)


def sigmoid(Z):
    """
    Sigmoid activation function
    Args:
        Z: array of Z values(output of linear function)

    Returns: array of A values, and caches the Z array for back prop

    """


    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache
def relu(Z):
    """
    relu activation function

    Args:
        Z: array of Z values (output of linear function)

    Returns: array of A values, caches the Z array

    """

    A = np.maximum(0,Z)  
    cache = Z 
    return A, cache
def sigmoid_backward(dA, cache):
    """
    back prop for sigmoid function unit

    Args:
        dA: array of dA values
        cache: cache of Z array

    Returns: returns array of dZ

    """

    Z = cache 
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
def relu_backward(dA, cache):
    """
     back prop for a relu function unit
    Args:
        dA: array of dA values
        cache: cache of Z array

    Returns: return array of dZ

    """
    Z = cache
    # just converting dz to a correct object.
    dZ = np.array(dA, copy=True)
    # When z <= 0, we should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    return dZ


def initialize_parameters_deep(layer_dimensions):
    """
    Initilizze the parameters
    Args:
        layer_dimensions: list of layer dimensions

    Returns: parameter dictionary with key being name of parameter, and values being the array of values

    """
    parameters = {}
    
    # number of layers in the network
    L = len(layer_dimensions) 

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dimensions[l], 1))
        
    return parameters



#forward propogation linear (calculate z, and cache the A,W and b valkues for activation function)
def forward_prop_linear(A, W, b):
    """
    linear forward prop through
    Args:
        A: array of prev A values
        W: array of weights for the layer
        b: array of biases for the layer

    Returns: return Z values, and caches A,W,b

    """

    Z = np.dot(W,A)+b

    cache = (A, W, b)
    
    return Z, cache


def forward_prop_activation(A_prev, W, b, activation):
    """
    forward prop with activation function
    Args:
        A_prev:  array of previous A values
        W: array of weights for the layer
        b: array of biases for the layer
        activation: string of activation function

    Returns: array of new A values and caches the A,W,b values, as well as the z values

    """

    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".       
        Z, linear_cache = forward_prop_linear(A_prev,W,b)
        A, activation_cache = sigmoid(Z)      
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".      
        Z, linear_cache = forward_prop_linear(A_prev,W,b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    implements forward prop through each layer
    Args:
        X: X values (input)
        parameters: output of initialize parametersß function

    Returns: array of final AL values, and cache dictionary with each item in the list being a layer

    """
    caches = []
    A = X

    # number of layers in the neural network
    L = len(parameters) // 2
    
    # Using a for loop to replicate [LINEAR->RELU] (L-1) times
    for l in range(1, L):
        A_prev = A 

        # Implementation of LINEAR -> RELU.
        A, cache = forward_prop_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")

        # Adding "cache" to the "caches" list.
        caches.append(cache)

    
    # Implementation of LINEAR -> SIGMOID.
    AL, cache = forward_prop_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    
    # Adding "cache" to the "caches" list.
    caches.append(cache)
            
    return AL, caches


def compute_cost(AL, Y):
    """
    computes the cost

    Args:
        AL:array of predicted AL values
        Y:array of true Y value

    Returns: cost

    """
    # number of examples
    m = Y.shape[1]

    # Compute loss from AL and y.
    cost = -1./m * np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))

    # To make sure our cost's shape is what we expect (e.g. this turns [[23]] into 23).
    cost = np.squeeze(cost)
    
    return cost


def back_prop_linear(dZ, cache):
    """
    linear back prop function
    Args:
        dZ: array of dZ values
        cache: caches of A_prev, W, and b

    Returns: arrays of dA_prev, dW, and db

    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def back_prop_activation(dA, cache, activation):
    """
    back prop for actiation functions
    Args:
        dA: array of dA values
        cache: cache of linear_cache and activation cache
        activation:

    Returns: array of dA_prev, dW, db


    """

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = back_prop_linear(dZ, cache[0])
   
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache[1])
        dA_prev, dW, db = back_prop_linear(dZ, cache[0])
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    backprop implementation through each layer
    Args:
        AL: array of final AL
        Y: array true Y values
        caches: list of caches of each layer

    Returns: a dictionary with each gradient value for each layer


    """
    grads = {}

    # the number of layers
    L = len(caches)
    m = AL.shape[1]

    # after this line, Y is the same shape as AL
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = back_prop_activation(dAL, current_cache, "sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". 
        # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = back_prop_activation(grads["dA"+str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads




def update_parameters(parameters, grads, learning_rate):
    """
    update parameters
    Args:
        parameters: dictionary of parameters
        grads: dictionary of gradients
        learning_rate: learning rate

    Returns:  dictionary of new parameters

    """
	# number of layers in the neural network
    L = len(parameters) // 2 

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate*grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate*grads["db" + str(l+1)])

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 1.0, num_iterations = 10, print_cost=False):
    """
    fully implemented model with L layers

    Args:
        X: input X values (array)
        Y: true Y values (array)
        layers_dims: neural net dimensions (list)
        learning_rate: learning rate (flaot)
        num_iterations: number of iteraitons (int)
        print_cost: print cost or not (boolean)

    Returns: parameter array, and list of costs through each iteration


    """
    # keep track of cost
    costs = []
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        print("iteration number", i)

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    print(costs)
    
    return parameters,costs



def predict(X, parameters):
    """
    predict Y values
    Args:
        X: input X array
        parameters: parameter array

    Returns: array of predicted Y values

    """
    m = X.shape[1]

    # number of layers in the neural network
    n = len(parameters) // 2
    p = np.zeros((1,m))
    
    # Forward propagation
    params, caches = L_model_forward(X, parameters)
    print(params)

    # convert params to 0/1 predictions
    for i in range(0, params.shape[1]):
        if params[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0   
        
    return p



parameters, costs = L_layer_model(X_train, y_train, layers_dims = [10000,20,10,1],learning_rate = 0.1,num_iterations = 3000, print_cost = True)
predictions = predict(X_test,parameters)


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_train.reshape(1247,1),predictions.reshape(1247)))
print(classification_report(y_test.reshape(23699,1),predictions.reshape(23699)))







