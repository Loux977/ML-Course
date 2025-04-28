import matplotlib.pyplot as plt
import numpy as np
from scratch_libraries.evaluation_metrics import RegressionMetrics

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(sig):
    return sig * (1 - sig)


class RegressionNeuralNetwork:
    def __init__(self, layers, epochs=700, alpha=1e-2, lmd=1, seed=42):
        self.layers = layers
        self.n_layers = len(layers)
        self.epochs = epochs
        self.alpha = alpha
        self.lmd = lmd
        self.seed =seed

        self.W = {}
        self.b = {}
        self.loss = []

    def init_parameters(self):
        np.random.seed(self.seed)
        for l in range(1, self.n_layers):
            self.W[l] = np.random.randn(self.layers[l], self.layers[l - 1])
            self.b[l] = np.ones((self.layers[l], 1))

    def forward_propagation(self, X):
        values = {}
        for l in range(1, self.n_layers):
            if l == 1:
                values["Z" + str(l)] = np.dot(self.W[l], X) + self.b[l]
            else:
                values["Z" + str(l)] = np.dot(self.W[l], values["A" + str(l - 1)]) + self.b[l]
            
            if l == self.n_layers - 1: 
                # just for the output layer we don't apply the sigmoid activation function
                values["A" + str(l)] = values["Z" + str(l)]
            else:
                # instead we apply sigmoid to hidden layers
                values["A" + str(l)] = sigmoid(values["Z" + str(l)])
        return values


    def compute_cost(self, y, values):
        y_pred = values["A" + str(self.n_layers - 1)] # the values in output from the last layer A^[L]=Z^[L]
        # Compute the Mean Squared Error (MSE) loss
        cost = 1/2 * np.average((y_pred - y) ** 2)
        
        m = y.shape[1]
        reg_sum = 0
        for l in range(1, self.n_layers):
            reg_sum += np.sum(np.square(self.W[l]))
        L2_reg = (self.lmd / (2 * m)) * reg_sum

        return cost + L2_reg


    def compute_cost_derivative(self, y, values):
        # Closed-form derivative of the MSE loss
        return values - y


    def backpropagation_step(self, values, X, y):
        m = y.shape[1]

        grads = {}

        dZ = None
        for l in range(self.n_layers - 1, 0, -1):
            if l == (self.n_layers - 1):
                dA = self.compute_cost_derivative(y, values["A" + str(l)])
                dZ = dA # set dZ=dA (no sigmoid is used)
            else:
                dA = np.dot(self.W[l + 1].T, dZ)
                dZ = np.multiply(dA, sigmoid_derivative(values["A" + str(l)]))

            if l == 1:
                grads["W" + str(l)] = (1 / m) * (np.dot(dZ, X.T) + self.lmd * self.W[l])
            else:
                grads["W" + str(l)] = (1 / m) * (np.dot(dZ, values["A" + str(l - 1)].T) + self.lmd * self.W[l])

            grads["b" + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        return grads

    def update(self, grads):
        for l in range(1, self.n_layers):
            self.W[l] -= self.alpha * grads["W" + str(l)]
            self.b[l] -= self.alpha * grads["b" + str(l)]

    def fit(self, X_train, y_train):
        # transpose X_train and y_train to align with our derivation
        X_train = X_train.T
        y_train = y_train.T

        self.init_parameters()
        for _ in range(self.epochs):
            values = self.forward_propagation(X_train)
            grads = self.backpropagation_step(values, X_train, y_train)
            self.update(grads)

            cost = self.compute_cost(y_train, values)
            self.loss.append(cost)

    def predict(self, X_test):
        # transpose X_test to align with our derivation
        X_test = X_test.T

        values = self.forward_propagation(X_test)
        y_pred = values["A" + str(self.n_layers - 1)]
        return y_pred.T # here the trasponse is used to return back to our original notation

    def compute_performance(self, y_test, y_pred):
        return RegressionMetrics(self).compute_performance(y_test, y_pred)

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss curve")
        plt.show()
