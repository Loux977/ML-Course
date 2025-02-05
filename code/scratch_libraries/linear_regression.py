import numpy as np

class LinearRegression:
    """
    Class to model a linear regression. This class has all the methods to be trained with different strategies
    and one method to produce a full prediction based on input samples. Moreover, this one is equipped by one method to
    measure performance and another method to build learning curves
    """
    def __init__(self, learning_rate=1e-2, n_steps=200, n_features=1, lmd=0.01, seed=123):
        """
        :param learning_rate: learning rate value
        :param n_steps: number of epochs for the training
        :param n_features: number of features involved in the regression
        :param lmd: regularization factor -> lmd_ is an array useful when is necessary compute theta's update with regularization factor
        """
        # Set a seed for random number generation to ensure reproducibility
        self.seed = seed
        np.random.seed(self.seed)

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features) # column vector (shape n+1 x 1) # +1 is due to theta_0
        self.lmd = lmd

        self.lmd_ = np.zeros(n_features)
        self.lmd_ = np.full(n_features, lmd)
        self.lmd_[0] = 0

    def fit_fbgd(self, X_train, y_train):
        """
        apply full batch gradient descent, without regularization, to the training set and return the evolution
        history of train and validation costs.
        :param X_train: training samples with bias of shape (m x n+1) # +1 is due to x_0
        :param y_train: training target values (shape m x 1)
        :return: history of evolution about cost and theta during training steps and, cost during validation phase
        """
        m=len(X_train) # number of training samples
        # initilize cost history and theta history values
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = np.dot(X_train, self.theta) # Xθ
            error = preds - y_train # Xθ-y
            # in matrix form: ∇J(θ) = 1/m X^T(Xθ-y)
            gradient = 1/m * np.dot(X_train.T, error)
            self.theta = self.theta - self.learning_rate * gradient # GD update rule

            # Just for logging
            theta_history[step, :] = self.theta.T # here we store theta as a row vector
            cost_history[step] = 1/(2*m) * np.dot(error,error.T) # in matrix form: J(θ) = 1/2m (Xθ-y)^2 = 1/2m (Xθ-y)^T * (Xθ-y)           

        return cost_history, theta_history

    def fit_regularized_fbgd(self, X_train, y_train):
        m=len(X_train)
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = np.dot(X_train, self.theta)
            error = preds - y_train
            self.theta = self.theta - self.learning_rate * 1/m * (np.dot(X_train.T, error) + self.lmd_ * self.theta) # + derivative of l2 regularization

            # Just for logging
            cost_history[step]= 1/(2*m) * (np.dot(error,error.T) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:])) # + l2 regularization            
            theta_history[step, :] = self.theta.T
        
        return cost_history, theta_history


    def fit_sgd(self, X_train, y_train):
        m=len(X_train)
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(self.n_steps):
            random_index = np.random.randint(m)
            x_i=X_train[random_index]
            y_i=y_train[random_index]

            pred = np.dot(x_i, self.theta)
            error = pred - y_i
            self.theta = self.theta - self.learning_rate * x_i.T * error

            # Just for logging
            pred = np.dot(X_train, self.theta)
            error_train = pred - y_train
            cost_history[step] = 1/(2*m) * np.dot(error_train, error_train.T)            
            #cost_history[step] = (1/(2*m))*np.sum(error_train)**2
            theta_history[step, :] = self.theta.T

        return cost_history, theta_history

    def fit_sgd_v2(self, X_train, y_train):
        m=len(X_train)
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for epoch in range(self.n_steps):
            for i in range(m):
                prediction = np.dot(X_train[i], self.theta)
                error = prediction - y_train[i]
                self.theta = self.theta - self.learning_rate * X_train[i].T * error
                theta_history[epoch, :] = self.theta.T

            # Just for logging
            pred = np.dot(X_train, self.theta)
            error_train = pred - y_train
            cost_history[epoch] = 1/(2*m) * np.dot(error_train, error_train.T)            
            #cost_history[step] = (1/(2*m))*np.sum(error_train)**2
            theta_history[epoch, :] = self.theta.T

        return cost_history, theta_history


    def fit_mbgd(self, X_train, y_train, batch_size=4):
        m = len(X_train)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(self.n_steps):
            # Select a single batch of samples uniformly at random
            indices = np.random.choice(m, batch_size)
            x_b = X_train[indices]
            y_b = y_train[indices]

            # Compute predictions and errors for the batch
            pred_b = np.dot(x_b, self.theta)
            error_b = pred_b - y_b

            # Update parameters using the gradient from the batch
            self.theta = self.theta - self.learning_rate * (1 / batch_size) * np.dot(x_b.T, error_b)

            # Logging for cost and theta history
            pred_train = np.dot(X_train, self.theta)
            error_train = pred_train - y_train
            cost_history[step] = 1 / (2 * m) * np.dot(error_train, error_train.T)
            theta_history[step, :] = self.theta.T

        return cost_history, theta_history


    def fit_mbgd_v2(self, X_train, y_train, batch_size=4):
        m = len(X_train)
        cost_history = np.zeros(self.n_steps)
        theta_history= np.zeros((self.n_steps, self.theta.shape[0]))

        for epoch in range(self.n_steps):
            for i in range (0, m, batch_size):
                x_b = X_train[i:i+batch_size]
                y_b = y_train[i:i+batch_size]

                pred_b = np.dot(x_b, self.theta)
                error_b = pred_b - y_b
                self.theta = self.theta - self.learning_rate * (1/len(x_b)) * np.dot(x_b.T, error_b)

            # Just for logging                
            pred_train = np.dot(X_train, self.theta)
            error_train = pred_train - y_train
            cost_history[epoch]= (1/(2*m)* np.dot(error_train, error_train.T))
            theta_history[epoch, :] = self.theta.T

        return cost_history, theta_history


    def predict(self, X_test):
        """
        perform a complete prediction on X samples
        :param X_test: test sample with shape (m, n_features)
        :return: prediction wrt X sample. The shape of return array is (m,)
        """
        return np.dot(X_test, self.theta)
