import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, n_features, n_steps, random_state=123):
        # Set a seed for random number generation to ensure reproducibility
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.randn(n_features)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit_full_batch(self, x, y):
        m = len(x)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(self.n_steps):
            # in matrix form: ∇J(θ) = 1/m X^T(h_θ-y)
            z = np.dot(x, self.theta) # z = Xθ
            prediction = self.sigmoid(z) # h_θ = g(z)
            error = prediction - y # h_θ-y
            gradient = 1/m * np.dot(x.T, error)

            self.theta = self.theta - self.learning_rate * gradient  # GD Update rule

            theta_history[step, :] = self.theta.T
            # in matrix form: J(θ) = -1/m [(y * log(⁡h_θ(x)) +(1-y) * log⁡(1 - h_θ(x))] 
            cost_history[step] = - (1/m) * (np.dot(y, np.log(prediction)) + np.dot(1-y, np.log(1-prediction)))

        return cost_history, theta_history

    def predict(self, x, threshold=0.5):
        z = np.dot(x, self.theta)
        prediction = self.sigmoid(z)
        return prediction > threshold
