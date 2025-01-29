import matplotlib.pyplot as plt
import numpy as np
from scratch_libraries.metrics import ClassificationMetrics


# Define the sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the derivative of the sigmoid function
def sigmoid_derivative(sig):
    # This approach assumes sig (i.e. the sigmoid activation a^[l]) is already computed during the forward pass 
    return sig * (1 - sig)
    # It is equivalent to the formula (a^[l] ⊙ (1 - a^[l])). Uncomment the line below for an explicit Hadamard implementation:
    #return np.multiply(sig, (1 - sig))  # the two are equivalent

# Define the NeuralNetwork class with specified parameters
class NeuralNetwork:
    # Initialize the neural network with layers, epochs, learning rate, and regularization parameter
    def __init__(self, layers, epochs=700, alpha=1e-2, lmd=1, seed=42):
        self.layers = layers
        self.n_layers = len(layers)
        self.epochs = epochs
        self.alpha = alpha
        self.lmd = lmd
        self.seed =seed

        # Initialize weights, biases, and loss variables
        self.W = {}
        self.B = {}
        self.loss = []
        self.loss_val = []

    # Initialize weights and biases for each layer randomly
    def init_parameters(self):
        # Set a seed for random number generation to ensure reproducibility
        np.random.seed(self.seed)
        for l in range(1, self.n_layers):
            self.W[l] = np.random.randn(self.layers[l], self.layers[l - 1]) # self.W[l] is a matrix of shape (units in layer l, units in layer l−1).
            self.B[l] = np.ones((self.layers[l], 1)) # self.B[l] is a column vector of shape (units in layer l, 1)

    # Perform forward propagation through the neural network
    def forward_propagation(self, X):
        # Initialize a dictionary to store intermediate values (i.e all Z and A) during forward propagation
        # (these values will be useful to compute partial derivates during backprop)
        values = {}
        # Iterate through all layers of the neural network
        for l in range(1, self.n_layers):
            if l == 1:
                # Compute the weighted sum for the first layer (considering X values)
                values["Z" + str(l)] = np.dot(self.W[l], X) + self.B[l] # we save Z^[l] values
                # IMPLICIT BROADCASTING: Here the bias vector for layer l has shape (d^[l], 1) and it's automatically broadcasted
                # to match the shape (d^[l], m), so that it can be added to the result of np.dot(self.W[l], X).
            else:
                # Compute the weighted sum for subsequent layers (considering A values of previous layer)
                values["Z" + str(l)] = np.dot(self.W[l], values["A" + str(l - 1)]) + self.B[l]

            # Apply the sigmoid activation function
            values["A" + str(l)] = sigmoid(values["Z" + str(l)]) # we save A^[l] values

        return values

    # Compute the cost function with L2 regularization
    def compute_cost(self, values, y):
        
        # Extract the predicted values , i.e. the values in output from the last layer A^[L]
        pred = values["A" + str(self.n_layers - 1)]
        # Compute the Binary Cross Entropy (BCE) loss
        cost = -np.average(y * np.log(pred) + (1 - y) * np.log(1 - pred)) # here np.average directly sum up all values from 1 to m and then divide the result by 1/m
        
        m = y.shape[0] # number of training examples
        # Compute the L2 regularization term
        reg_sum = 0
        for l in range(1, self.n_layers):
            reg_sum += np.sum(np.square(self.W[l]))
        L2_reg = (self.lmd / (2 * m)) * reg_sum

        # Return the total cost including regularization
        return cost + L2_reg

    # Compute the derivative of the cost function
    def compute_cost_derivative(self, values, y):
        # Closed-form derivative of the BCE loss
        return -(np.divide(y, values) - np.divide(1 - y, 1 - values))

    # Perform a single backpropagation step and update parameters
    def backpropagation_step(self, values, X, y):
        m = X.shape[1] # number of training examples

        # Initialize a dictionary to store the parameter updates (i.e all W and B)
        params_upd = {}

        # Initialize the derivative of the weighted sum
        dZ = None
        # Iterate backward through the layers of the neural network
        for l in range(self.n_layers - 1, 0, -1):
            if l == (self.n_layers - 1):
                # For the output layer, compute the derivative of the cost function
                dA = self.compute_cost_derivative(values["A" + str(l)], y)
            else:
                # For hidden layers, compute the derivative using the chain rule
                dA = np.dot(self.W[l + 1].T, dZ) # dZ here is the error coming from the next layer (i.e. 𝛿^[l+1])

            # Compute the derivative of the weighted sum
            dZ = np.multiply(dA, sigmoid_derivative(values["A" + str(l)]))

            if l == 1:
                # Compute the weight gradients for the first layer
                params_upd["W" + str(l)] = (1 / m) * (np.dot(dZ, X.T) + self.lmd * self.W[l])
            else:
                # Compute the weight gradients for subsequent layers
                params_upd["W" + str(l)] = (1 / m) * (np.dot(dZ, values["A" + str(l - 1)].T) + self.lmd * self.W[l])

            # Compute the bias gradients (regularization is not applied to bias)
            params_upd["B" + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            # Note: Here np.sum(dZ, axis=1) computes the sum of dZ (𝛿[l]) across all m training examples (axis=1) for each of
            # the d unit in layer l (i.e. dz_1_[l], dz_2_[l], ..., dz_d_[l]). This results in a vector of shape (d^[l],) where each element
            # corresponds to the total error for a single unit. keepdims=True ensures the result has shape (d^[l],1), matching the shape of self.B[i].

        return params_upd

    # GD Update Rule: Update weights and biases based on the calculated gradients
    def update(self, upd):
        for l in range(1, self.n_layers):
            self.W[l] -= self.alpha * upd["W" + str(l)]
            self.B[l] -= self.alpha * upd["B" + str(l)]

    # Train the neural network on the provided data
    def fit(self, X_train, y_train):
        self.loss = []
        self.loss_val = []
        self.init_parameters()

        # transpose X_train and y_train to align with our derivation
        X_train = X_train.T
        y_train = y_train.T

        for _ in range(self.epochs):
            # Perform forward and backward passes, and update the parameters
            values = self.forward_propagation(X_train)
            grads = self.backpropagation_step(values, X_train, y_train)
            self.update(grads)

            # Compute and record the training loss
            cost = self.compute_cost(values, y_train)
            self.loss.append(cost)

    # Make predictions on new data
    def predict(self, X_test):
        # transpose X_test to align with our derivation
        X_test = X_test.T

        values = self.forward_propagation(X_test)
        pred = values["A" + str(self.n_layers - 1)]
        return np.round(pred.T) # here the trasponse is used to return back to our original notation

    # Compute classification performance metrics
    def compute_performance(self, preds, y):
        return ClassificationMetrics(self).compute_performance(preds, y)

    # Plot the training loss curve
    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss curve")
        plt.show()