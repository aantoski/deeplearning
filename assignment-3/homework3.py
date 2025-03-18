import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 1
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ 74 ] # number of neurons in hidden layers
NUM_OUTPUT = 10

def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i+1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

def relu(z):
    return np.maximum(0.0, z)

def d_relu(Z):
    # if isinstance(Z, np.ndarray):
    return (Z > 0).astype(float)
    # else:
    #     print("Z value: ", Z)
    #     raise ValueError("Input to d_relu must be a NumPy array")

def softmax(z):
    z_max = np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z - z_max)  # Stability trick
    softmax_z = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    return softmax_z

def fCE (X, Y, weights):
    Ws, bs = unpack(weights)
    n = X.shape[1]
    active, pre_active = forward(X, weights)
    epsilon = 1e-8  # Small constant to avoid log(0)

    ce = (-1 / n) * np.sum(Y.T * np.log(active[-1].T + epsilon), axis=1)
    alpha = 0.00001 
    
    # L2 regularization term
    l2_penalty = (alpha / 2 * n) * sum(np.sum(W ** 2) for W in Ws)
    
    total_loss = np.mean(ce[0]) + l2_penalty
    
    return ce[0]

def fCE_unreg (X, Y, weights):
    n = X.shape[0]
    active, pre_active = forward(X, weights)
    epsilon = 1e-8  # Small constant to avoid log(0)
    ce = (-1 / n) * np.sum(Y.T * np.log(active[-1].T + epsilon), axis=1)
     
    return active[-1], ce[0]

def forward(X, weights):
    Ws, bs = unpack(weights)    
    active = [X] # X and H
    pre_active = [] # Z
    layers = len(Ws)

    for i in range(layers):
        b = bs[i].reshape(-1,1)
        Z = Ws[i].dot(active[-1]) + b
        
        pre_active.append(Z)
        
        if i == layers - 1:
            h = softmax(Z)
        else:
            h = relu(Z)
            # print("Values of Z ", Z)
            # print("Values of h (after relu): ", h)
        
        active.append(h)

    return active, pre_active
   
def gradCE (X, Y, weights):   
    active, preactive = forward(X, weights)
    Ws, bs = unpack(weights)
    layers = len(Ws)
    n = X.shape[1]
    allGradientsAsVector = []
    weightsVector = []
    biasVector = []
    g = active[-1] - Y
    
    # print("Preactive values (Z): ")
    # print(active)
    
    # print("Active values (h): ")
    # print(preactive)
    
    # print("Length of active: ", len(active))
    # print("Length of preactive: ", len(preactive))
    
    # print("gradCE X shape: ", X.shape)
    # print("gradCE Y shape: ", Y.shape)
    
    for k in reversed(range(layers)):
        if k != layers - 1:  
            second = d_relu(preactive[k])
            # print("Preactive: ", second)
            g = g * second
        
        grad_W_k = (g.dot(active[k].T) / n) + Ws[k] * (.000001 / n)
        grad_b_k = np.sum(g, axis=1, keepdims=True) / n
        
        weightsVector = np.concatenate([weightsVector, grad_W_k.flatten()])
        biasVector = np.concatenate([biasVector, grad_b_k.flatten()])
        
        if k > 0:
            W = Ws[k].T
            g = W.dot(g)
    
    allGradientsAsVector = np.concatenate([weightsVector, biasVector])
    
    return allGradientsAsVector

def backward(X, Y, weights):
    # active, preactive = forward(X, weights)
    Ws, bs = unpack(weights)
    layers = len(Ws)
    n = X.shape[0]
    allGradientsAsVector = []
    batch_size = 128
    num_batches = n // batch_size
        
    alpha = 0.00001  # L2 regularization parameter
    
    # To accumulate gradients across all batches
    grad_W_total = [np.zeros_like(W) for W in Ws]
    grad_b_total = [np.zeros_like(b) for b in bs]
    
    for i in range(0, n, batch_size):
        print(f"Processing batch {i // batch_size + 1} out of {n // batch_size + 1}")
        X_batch = X[i:i+batch_size, :]
        Y_batch = Y[:, i:i+batch_size]
        
        active, preactive = forward(X_batch, weights)
        g = active[-1] - Y_batch.T
        
        # print("Initial g shape: ", g.shape)
    
        for k in range(layers - 1, -1, -1):
            # print("Backward prop layer: ", k)
            # print("Bias shape: ", bs[k].shape)
            # Calculate g
            if k < layers - 1:  # hidden layers
                second = d_relu(preactive[k])
                g *= second
                # print("Updated g shape (hidden): ", g.shape)
            
            if k == 0: # input layer
                grad_W_batch = (g.T.dot(X_batch) / batch_size) + (alpha / batch_size) * Ws[k]
            else:
                active_h = active[k-1]
                grad_W_batch = (g.T.dot(active_h) /batch_size) + (alpha / batch_size) * Ws[k]
            
            grad_b_batch = np.sum(g, axis=0) / batch_size
            
            # Accumulate gradients from each batch
            grad_W_total[k] += grad_W_batch
            grad_b_total[k] += grad_b_batch
            
            # propagate new g
            if k > 0:
                g = g.dot(Ws[k])
                
    allGradientsAsVector = np.concatenate([grad_W.flatten() for grad_W in grad_W_total] + 
                                        [grad_b.flatten() for grad_b in grad_b_total])
    
    return allGradientsAsVector / num_batches
        
# Creates an image representing the first layer of weights (W0).
def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()
    
def visualize_W1 (W):
    Ws,bs = unpack(W)
    W = Ws[1]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()

def accuracy(y_true, y_pred):
    y_true = np.argmax(y_true, axis=0)
    y_pred = np.argmax(y_pred, axis=0)
    correct_predictions = np.sum(y_pred == y_true)
    return correct_predictions / len(y_true) * 100


def sgd(X, y, weights, lr, batch_size, epochs, alpha):
    Ws, bs = unpack(weights)
    n = X.shape[1]
    
    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            X_batch = X[:, i:i+batch_size]
            y_batch = y[:, i:i+batch_size]
            
            allGradients = gradCE(X_batch, y_batch, weights)
            
            grad_Ws, grad_bs = unpack(allGradients)
            
            # update Ws and bs using the gradients
            for j in range(len(Ws)):
                Ws[j] -= lr * grad_Ws[j]
                bs[j] -= lr * grad_bs[j]
            
            # Ws = [W - lr * grad_W for W, grad_W in zip(Ws, grad_Ws)]
            # bs = [b - lr * grad_b for b, grad_b in zip(bs, grad_bs)]
        
        # Check accuracy every 10 epochs
        if (epoch + 1) % 5 == 0:
            updated_weights = np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])
            y_pred, _ = fCE_unreg(X, y, updated_weights)
            acc = accuracy(y, y_pred)
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {acc:.2f}%")
    
    updated_weights = np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])
    
    return updated_weights
 
 
def train (trainX, trainY, weights, testX, testY):
    # Split training set to create a validation set.
    permutation = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, permutation]
    trainY = trainY[:, permutation]
    
    split_index = int(trainX.shape[1] * 0.85)
    
    trainX, valX = trainX[:, :split_index], trainX[:, split_index:]
    trainY, valY = trainY[:, :split_index], trainY[:, split_index:]
    
    learning_rates = [.1, .01, .001]
    epochs = [20,30,50]
    batch_sizes = [128, 256]
    alphas = [.0001, .001]
    
    best_ce = float('inf')
    best_params = {}
    
    # Start with mini batches to update the weights and biases.
    for lr in learning_rates:
        for epoch in epochs:
            for batch_size in batch_sizes:
                for a in alphas: 
                    print(f"Running with learning rate: {lr}, epochs: {epoch}, batch size: {batch_size}, alpha: {a}")
                    updated_weights = sgd(trainX, trainY, weights, lr, batch_size, epoch, a)
                    
                    y_pred, ce = fCE_unreg(valX, valY, updated_weights)
                    acc = accuracy(valY, y_pred)
                    ce = np.mean(ce)
                    
                    print("Unregularized CE: ", ce)
                    print("Accuracy: ", acc)
                    
                    print("=====================================\n")
                    
                    # Save best parameters
                    if ce < best_ce:
                        best_ce = ce
                        best_params = {'learning_rate': lr, 'epochs': epoch, 'batch_size': batch_size, 'parameters': updated_weights, 'alpha': a}
                        best_weights = updated_weights
                        print("Updated weights and bias")
                    
    # TODO: add an evaluate function outside of train to evaluate the model on the test set.
    print("Best ce: ", best_ce)
    print("Best parameters: ", best_params)

    y_pred, ce = fCE_unreg(testX, testY, best_weights)
    acc = accuracy(testY, y_pred)
    
    print("Accuracy of best model: ", acc)
    
    # Return the weights of the best model
    return updated_weights

def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight using a variant of the Kaiming He Uniform technique.
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN[i+1], NUM_HIDDEN[i]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1])
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1]))/NUM_HIDDEN[-1]**0.5) - 1./NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs

# One hot encode labels
def one_hot_encode(y_labels):
    y_labels = y_labels.astype(int)
    one_hot = np.zeros((10, len(y_labels)))
    one_hot[y_labels, np.arange(len(y_labels))] = 1
    return one_hot

if __name__ == "__main__":
    # Load training data.
    trainX = np.load('/Users/antoski/WPI/Deep Learning/Homework_3/trainX.npy').T
    trainY = np.load('/Users/antoski/WPI/Deep Learning/Homework_3/trainY.npy')
    testX = np.load('/Users/antoski/WPI/Deep Learning/Homework_3/testX.npy').T
    testY = np.load('/Users/antoski/WPI/Deep Learning/Homework_3/testY.npy')

    # Print shapes of loaded data
    print(f"trainX shape: {trainX.shape}")
    print(f"trainY shape: {trainY.shape}")
    print(f"testX shape: {testX.shape}")
    print(f"testY shape: {testY.shape}")

    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).
    trainX = trainX / 255.0 - 0.5
    testX = testX / 255.0 - 0.5
    
    # One-hot encoded y
    one_hot_trainY = one_hot_encode(trainY)
    one_hot_testY = one_hot_encode(testY)

    # Print shapes of one-hot encoded labels
    print(f"one_hot_trainY shape: {one_hot_trainY.shape}")
    print(f"one_hot_testY shape: {one_hot_testY.shape}")

    # TODO: loop for different layer and neuron counts
    Ws, bs = initWeightsAndBiases()
    
    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

    # Print shapes of initialized weights and biases
    for i, (W, b) in enumerate(zip(Ws, bs)):
        print(f"W{i} shape: {W.shape}")
        print(f"b{i} shape: {b.shape}")
    
    # On just the first 5 training examlpes, do numeric gradient check.
    # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # The lambda expression is used so that, from the perspective of
    # check_grad and approx_fprime, the only parameter to fCE is the weights
    # themselves (not the training data).
    
    # sample_size = 1
    
    # # Select the first training example
    # trainX_example = np.atleast_2d(trainX[:, :sample_size])
    # trainY_example = np.atleast_2d(one_hot_trainY[:, :sample_size])
    
    # # print("TrainX example shape:", trainX_example.shape)
    # # print("TrainY example shape: ", trainY_example.shape)
    
    # # Define lambda functions for fCE and gradCE
    # fCE_lambda = lambda weights_: fCE(trainX_example, trainY_example, weights_)
    # gradCE_lambda = lambda weights_: gradCE(trainX_example, trainY_example, weights_)

    # # Check the gradient
    # gradient_difference = scipy.optimize.check_grad(fCE_lambda, gradCE_lambda, weights)
    # print("Gradient difference:", gradient_difference)

    # # Compute numerical gradient
    # numerical_gradient = scipy.optimize.approx_fprime(weights, fCE_lambda, 1e-5)
    # print("Numerical gradient:", numerical_gradient)

    # # Compute analytical gradient
    # analytical_gradient = gradCE(trainX_example, trainY_example, weights)
    # print("Analytical gradient:", analytical_gradient)

    # # # Unpack gradients for better comparison
    # num_grad_Ws, num_grad_bs = unpack(numerical_gradient)
    # ana_grad_Ws, ana_grad_bs = unpack(analytical_gradient)
    
    # # for i in enumerate(num_grad_Ws):
    # #     print(i)

    # # Print analytical gradients for weights
    # for i, (ana_grad_W, num_grad_W) in enumerate(zip(ana_grad_Ws, num_grad_Ws)):
    #     print(f"Analytical gradient W{i} shape:", ana_grad_W.shape)
    #     print(f"Analytical gradient W{i}:", ana_grad_W)
    #     print(f"Numerical gradient W{i} shape:", num_grad_W.shape)
    #     print(f"Numerical gradient W{i}:", num_grad_W)

    # # Print analytical gradients for biases
    # for i, (ana_grad_b, num_grad_b) in enumerate(zip(ana_grad_bs, num_grad_bs)):
    #     print(f"Analytical gradient b{i} shape:", ana_grad_b.shape)
    #     print(f"Analytical gradient b{i}:", ana_grad_b)
    #     print(f"Numerical gradient b{i} shape:", num_grad_b.shape)
    #     print(f"Numerical gradient b{i}:", num_grad_b)

    # # #Compare gradients
    # print("Difference between numerical and analytical gradients:", np.linalg.norm(numerical_gradient - analytical_gradient))
    
    batch_size = 1
     
    print(scipy.optimize.check_grad(lambda weights_: fCE(np.atleast_2d(trainX[:,:batch_size]), np.atleast_2d(one_hot_trainY[:,:batch_size]), weights_), \
                                    lambda weights_: gradCE(np.atleast_2d(trainX[:,:batch_size]), np.atleast_2d(one_hot_trainY[:,:batch_size]), weights_), \
                                    weights))
    
    print(scipy.optimize.approx_fprime(weights, lambda weights_: fCE(np.atleast_2d(trainX[:,:batch_size]), np.atleast_2d(one_hot_trainY[:,:batch_size]), weights_), 1e-8))
    

    # weights = train(trainX, one_hot_trainY, weights, testX, one_hot_testY)
    # print("Done!")
    # Ws, bs = unpack(weights)
    # for i, (W, b) in enumerate(zip(Ws, bs)):
    #     print(f"W{i}: {W}")
    #     print(f"b{i}: {b}")
    # show_W0(weights)
