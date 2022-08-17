import numpy as np
import matplotlib.pyplot as plt
from activations import *
import math

np.random.seed(1)

def initialize_parameters_deep(layer_dims):
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def initialize_parameters_deep_he(layers_dims):

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
         
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

    Z, linear_cache = linear_forward(A_prev, W, b)  # This "linear_cache" contains (A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z) # This "activation_cache" contains "Z"

    elif activation == "relu":
        A, activation_cache = relu(Z)

    elif activation == "tanh":
        A, activation_cache = tanh(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters, activation, last_activation):

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation)
        caches.append(cache)
    
    # Last layer
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], last_activation)
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

def compute_cost(AL, Y, cost_func="mse"):

    m = Y.shape[1] # no. of samples

    # Compute loss from aL and y
    if cost_func == "log":
        cost = (-1/m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
    elif cost_func == "mse":
        cost = np.mean(np.square(AL-Y))*0.5
    
    cost = np.squeeze(cost) # To make sure the shape correct
    assert(cost.shape == ())
    return cost

def linear_backward(dZ, cache):
    # Here cache is "linear_cache" containing (A_prev, W, b) coming from the forward propagation in the current layer
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
      
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, cost_func, activation, last_activation):

    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # Y is now the same shape as AL
    
    # Initializing backpropagation
    if(cost_func=='log'):
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    if(cost_func=='mse'):
        dAL = (AL-Y)
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1] # Last Layer
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, last_activation)
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
     
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = activation)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
      
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, cost_func = "mse", activation = "relu", last_activation = None, he_init=False):
    
    if last_activation == None:
        last_activation = activation

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code
    if he_init == True:
        parameters = initialize_parameters_deep_he(layers_dims)
    else:
        parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters, activation, last_activation)
        
        # Compute cost.
        cost = compute_cost(AL, Y, cost_func)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, cost_func, activation, last_activation)
     
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
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def test_regression(X, y, parameters, activation, last_activation):
  
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    res = np.zeros((1,m))
    
    # Forward propagation
    res, caches = L_model_forward(X, parameters, activation, last_activation)
    mse = np.mean(np.square(res-y))*0.5
    
    print("MSE: ", mse)
    return mse

def predict_binary_classes(X, y, parameters, activation, last_activation):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probs, caches = L_model_forward(X, parameters, activation, last_activation)
    
    # convert probs to 0/1 predictions
    for i in range(0, probs.shape[1]):
        if probs[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
            
    print("Accuracy: "  + str(np.sum((p == y)/m)))       
    return p

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def L_layer_model_minib(X, Y,layers_dims,valid=False,valid_x=None,valid_y=None, optimizer='gd', learning_rate = 0.0007,he_init=False, 
                        mini_batch_size = 64, beta = 0.9,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_iterations = 10000,
                        activation='sigmoid',regularisation='none',print_cost = True,lambd=0.1,cost_func='mse'):

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []  
    validcosts=[]                     # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]
    batches=m//mini_batch_size                   # number of training examples
    
    # Initialize parameters
    # parameters = initialize_parameters(layers_dims)
    if(he_init):
        parameters=initialize_parameters_deep_he(layers_dims)
    else:
        parameters=initialize_parameters_deep(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_iterations):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches =  L_model_forward(X=minibatch_X,parameters= parameters,activation= activation, last_activation = activation)

            # Compute cost and add to the cost total
            cost_total += compute_cost(AL= a3, Y= minibatch_Y,cost_func= cost_func)

            # Backward propagation
            grads =  L_model_backward(AL= a3,Y= minibatch_Y,caches= caches,cost_func = cost_func, activation= activation,last_activation = activation)

            # Update parameters
            if optimizer == "gd" or optimizer=='none':
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / batches
        
        # Print the cost every 1000 epoch
        if print_cost and i % 1 == 0:
            if(valid==True):
                valid_err=predicterr(valid_x,valid_y,parameters=parameters,lambd=lambd,activation=activation,regularisation='none',cost_func=cost_func)
                validcosts.append(valid_err)
                print ("Cost after epoch %i: %f  %f" %(i, cost_avg,valid_err))
            else:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    if(valid==True):
        plt.plot(validcosts)
        plt.legend(["train", "validation"], loc ="upper right") 
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters