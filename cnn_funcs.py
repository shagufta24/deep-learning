import numpy as np

def zero_pad(X, pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)))
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    # Element-wise product between a_slice_prev and W.
    s = np.multiply(a_slice_prev,W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    b = np.squeeze(b)
    Z = Z + b
    return Z

def conv_forward(A_prev, W, b, hparameters):
    # Retrieve dimensions from A_prev's shape 
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    n_H = int((n_H_prev + 2*pad - f)/stride) + 1
    n_W = int((n_W_prev + 2*pad - f)/stride) + 1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]          # Select ith training example's padded activation
        for h in range(n_H):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = stride * h 
            vert_end = vert_start  + f
            
            for w in range(n_W):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                horiz_start = stride * w
                horiz_end = horiz_start + f
                
                for c in range(n_C):   # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    weights = W[:, :, :, c]
                    biases  = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    cache = (A_prev, W, b, hparameters)
    return Z, cache   

    def pool_forward(A_prev, hparameters, mode = "max"):
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        # Retrieve hyperparameters from "hparameters"
        f = hparameters["f"]
        stride = hparameters["stride"]
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))              
        
        for i in range(m):                         # loop over the training examples
            a_prev_slice = A_prev[i]
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                vert_start = stride * h 
                vert_end = vert_start + f
                
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    # Find the vertical start and end of the current "slice" (≈2 lines)
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    
                    for c in range (n_C):            # loop over the channels of the output volume
                        
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_slice_prev = a_prev_slice[vert_start:vert_end,horiz_start:horiz_end,c]
                        
                        # Compute the pooling operation on the slice. 
                        # Use an if statement to differentiate the modes. 
                        # Use np.max and np.mean.
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_slice_prev)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_slice_prev)
                        else:
                            print(mode+ "-type pooling layer NOT Defined")    
        
        cache = (A_prev, hparameters)
        assert(A.shape == (m, n_H, n_W, n_C))
        return A, cache