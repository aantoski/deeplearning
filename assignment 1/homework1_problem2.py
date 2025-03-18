import numpy as np
from sklearn.model_selection import train_test_split

def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")
    
    print("Data loaded")

    # Split training data
    split_index = int(0.8 * len(X_tr))
    X_train = X_tr[:split_index]
    y_train = ytr[:split_index]
    X_val = X_tr[split_index:]
    y_val = ytr[split_index:]
    
    print("Validation set created")

    # SGD
    def sgd(X, y, lr, epochs, batch_size):
        m, n = X.shape
        weights = np.zeros(n)
        bias = 0
        
        for epoch in range(epochs):
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Predictions using the current weights and bias
                y_pred = X_batch.dot(weights) + bias
                
                # Gradient calculation for weights and bias
                weight_gradient = X_batch.T.dot(y_pred - y_batch) / batch_size
                bias_gradient = np.sum(y_pred - y_batch) / batch_size
                
                # Update weights and bias
                weights -= lr * weight_gradient
                bias -= lr * bias_gradient
                
        return weights, bias

    # MSE based on function provided
    def mse(y_true, y_pred):
        mse = (1 / (2 * len(y_true))) * np.sum((y_pred - y_true) ** 2)
        return mse

    # Hyperparameter configuartions
    learning_rates = [0.001, 0.0015, 0.002]
    epochs = [200, 250, 300]
    batch_sizes = [20, 25, 30]

    # Default mse and placeholder for best params
    best_mse = float('inf')
    best_params = {}

    # Go through every combination of hyperparameters
    for lr in learning_rates:
        for epochs in epochs:
            for batch_size in batch_sizes:
                weights, bias = sgd(X_train, y_train, lr, epochs, batch_size)
                
                # Calculate MSE on validation
                y_val_pred = X_val.dot(weights) + bias
                mse = mse(y_val, y_val_pred)
                print("MSE: ", mse, " with ε: ", lr, ", Epoch:", epochs, ", ñ:", batch_size)
                
                # Save best parameters
                if mse < best_mse:
                    best_mse = mse
                    best_params = {'learning_rate': lr, 'epochs': epochs, 'batch_size': batch_size, 'weights': weights, 'bias': bias}
                    #print("Updated - Weights: ", weights, " Bias: ", bias)

    # Use the found best parameters to train on full training set and test on test set
    best_lr = best_params['learning_rate']
    best_epochs = best_params['epochs']
    best_batch_size = best_params['batch_size']

    final_weights, final_bias = sgd(X_tr, ytr, best_lr, best_epochs, best_batch_size)
    
    # MSE on training and test sets
    y_train_pred = X_tr.dot(final_weights) + final_bias
    y_test_pred = X_te.dot(final_weights) + final_bias

    mse_train = mse(ytr, y_train_pred)
    mse_test = mse(yte, y_test_pred)

    # Print!!
    print("Best Parameters:", best_params)
    print("Lowest Validation MSE:", best_mse)
    print("MSE on Training Set:", mse_train)
    print("MSE on Test Set:", mse_test)

train_age_regressor()
