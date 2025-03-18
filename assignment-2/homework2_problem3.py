import numpy as np

def train_mnist_softmax():
    # Load data
    X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 784))
    ytr = np.load("fashion_mnist_train_labels.npy")
    X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 784))
    yte = np.load("fashion_mnist_test_labels.npy")
    
    print("Data loaded")
    
    # Normalize pixel values
    X_tr = X_tr / 255
    X_te = X_te / 255
    
    print("Normalized pixel values")

    # Split training data
    split_index = int(0.8 * len(X_tr))
    X_train = X_tr[:split_index]
    y_train = ytr[:split_index]
    X_val = X_tr[split_index:]
    y_val = ytr[split_index:]
    
    print("Split training data into training and validation sets")
    
    # One hot encode labels
    def one_hot_encode(y_labels):
        one_hot = np.zeros((len(y_labels), 10))
        one_hot[np.arange(len(y_labels)), y_labels] = 1
        return one_hot
    
    # One hot encode training, validation and test labels
    y_train_en = one_hot_encode(y_train)
    y_val_en = one_hot_encode(y_val)
    y_test_en = one_hot_encode(yte)
    
    print("One hot encoded labels for training, validation, and test sets")
    
    # SGD
    def sgd(X, y, lr, epochs, batch_size, alpha):
        m, n = X.shape
        
        # Starting weight matrix with 0-mean Gaussian distribution
        weights = np.random.normal(0, 1 / np.sqrt(10), (784, 10))
        biases = np.zeros(10)
        
        print(weights)
        
        for epoch in range(epochs):
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                z = X_batch.dot(weights) + biases
                y_pred = softmax(z)
                
                # Gradient calculation for weights and bias
                difference = y_pred - y_batch
                weight_gradient = (X_batch.T).dot(difference) / batch_size + alpha * weights
                bias_gradient = np.sum(difference, axis=0) / batch_size
                
                # Update weights and bias
                weights -= lr * weight_gradient
                biases -= lr * bias_gradient
                
        return weights, biases

    # CE based on function provided
    def cross_entrophy(alpha, y_true, y_pred, weights):
        # to make this easier to read we will break up equation: -1/n(D)+a/2(S)
        # where D is the double summation of the equation and S is the summation of weight vectors
        # Clip y_pred to avoid log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10) # avoid NaN
        
        D = np.sum(y_true * np.log(y_pred))
        S = sum(np.dot(w.T, w) for w in weights)
        
        ce = (-1 / len(y_true)) * D + (alpha / 2) * S
        return ce
    
    # Normalized z_k values
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Prevent overflow with large exponents
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # CE without regularization
    def performance_ce(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10) # avoid NaN
        
        return (-1 / len(y_true)) * np.sum(y_true * np.log(y_pred))
    
    # Accuracy of predictions
    def accuracy(y_true, y_pred):
        y_labels = np.argmax(y_pred, axis=1)
        correct_predictions = np.sum(y_true == y_labels)
        return correct_predictions / len(y_true) * 100

    # Hyperparameter configuartions
    # Removed extra values to speed up run time
    learning_rates = [0.05]
    epochs = [200]
    batch_sizes = [100]
    alphas = [.001]

    # Default mse and placeholder for best params
    best_ce = float('inf')
    best_params = {}
    
    print("Beginning training")

    # Go through every combination of hyperparameters
    for lr in learning_rates:
        for epoch in epochs:
            for batch_size in batch_sizes:
                for a in alphas: 
                    weights, bias = sgd(X_train, y_train_en, lr, epoch, batch_size, a)
                    
                    # Calculate CE on validation
                    y_val_pred = softmax(X_val.dot(weights) + bias)
                    ce = cross_entrophy(a, y_val_en, y_val_pred, weights)
                    print("CE: ", ce, " with ε: ", lr, ", Epoch:", epoch, ", ñ:", batch_size, ", α:" , a)
                    
                    # Save best parameters
                    if ce < best_ce:
                        best_ce = ce
                        best_params = {'learning_rate': lr, 'epochs': epoch, 'batch_size': batch_size, 'weights': weights, 'bias': bias, 'alpha': a}
                        print("Updated weights and bias")

    # Use the found best parameters to train on full training set and test on test set
    best_lr = best_params['learning_rate']
    best_epochs = best_params['epochs']
    best_batch_size = best_params['batch_size']
    best_alpha = best_params['alpha']

    final_weights, final_biases = sgd(X_train, y_train_en, best_lr, best_epochs, best_batch_size, best_alpha)
    
    print("Best parameters found, ")
    
    # Evaluate
    y_test_pred = X_te.dot(final_weights) + final_biases
    test_ce = performance_ce(y_test_en, y_test_pred)
    test_accuracy = accuracy(yte, y_test_pred)

    # Print!
    print("Best Parameters:", best_params)
    print("Lowest Validation CE:", best_ce)
    print("CE on Test Set (unregularized):", test_ce)
    print("Accuracy on Test Set:", test_accuracy)

train_mnist_softmax()
