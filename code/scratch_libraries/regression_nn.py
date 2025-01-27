import matplotlib.pyplot as plt
import numpy as np
from libraries.metrics import RegressionMetrics

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(sig):
    return sig * (1 - sig)


class NeuralNetwork:
    def __init__(self, layers, epochs=700, alpha=1e-2, lmd=1, seed=42):
        self.layers = layers
        self.n_layers = len(layers)
        self.epochs = epochs
        self.alpha = alpha
        self.lmd = lmd
        self.seed =seed

        self.W = {}
        self.B = {}
        self.loss = []
        self.loss_val = []

    def init_parameters(self):
        np.random.seed(self.seed)
        for l in range(1, self.n_layers):
            self.W[l] = np.random.randn(self.layers[l], self.layers[l - 1])
            self.B[l] = np.ones((self.layers[l], 1))

    def forward_propagation(self, X):
        values = {}
        for l in range(1, self.n_layers):
            if l == 1:
                values["Z" + str(l)] = np.dot(self.W[l], X) + self.B[l]
            else:
                values["Z" + str(l)] = np.dot(self.W[l], values["A" + str(l - 1)]) + self.B[l]
            
            if l == self.n_layers - 1: 
                # just for the output layer we don't apply the sigmoid activation function
                values["A" + str(l)] = values["Z" + str(l)]
            else:
                # instead we apply sigmoid to hidden layers
                values["A" + str(l)] = sigmoid(values["Z" + str(l)])
        return values

    # Compute the cost function with L2 regularization
    def compute_cost(self, values, y):
        
        # Extract the predicted values , i.e. the values in output from the last layer Z^[L]
        pred = values["Z" + str(self.n_layers - 1)]
        # Compute the Mean Squared Error (MSE) loss
        cost = 1/2 * np.average((pred - y) ** 2)
        
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
        # Closed-form derivative of the MSE loss
        return values - y


    def backpropagation_step(self, values, X, y):
        m = X.shape[1]

        params_upd = {}

        dZ = None
        for l in range(self.n_layers - 1, 0, -1):
            if l == (self.n_layers - 1):
                dA = self.compute_cost_derivative(values["A" + str(l)], y)
                dZ = dA # set dZ=dA (no sigmoid is used)
            else:
                dA = np.dot(self.W[l + 1].T, dZ)
                dZ = np.multiply(dA, sigmoid_derivative(values["A" + str(l)]))

            if l == 1:
                params_upd["W" + str(l)] = (1 / m) * (np.dot(dZ, X.T) + self.lmd * self.W[l])
            else:
                params_upd["W" + str(l)] = (1 / m) * (np.dot(dZ, values["A" + str(l - 1)].T) + self.lmd * self.W[l])

            params_upd["B" + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        return params_upd

    def update(self, upd):
        for l in range(1, self.n_layers):
            self.W[l] -= self.alpha * upd["W" + str(l)]
            self.B[l] -= self.alpha * upd["B" + str(l)]

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

    def predict(self, X_test):
        # transpose X_test to align with our derivation
        X_test = X_test.T

        values = self.forward_propagation(X_test)
        pred = values["A" + str(self.n_layers - 1)]
        return np.round(pred)

    def compute_performance(self, preds, y):
        clf_metrics = RegressionMetrics(self)
        return clf_metrics.compute_performance(preds.squeeze(), y)

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss curve")
        plt.show()
