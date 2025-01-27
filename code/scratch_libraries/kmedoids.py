from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KMedoids(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        """
        Initializes the KMedoids clustering instance.

        Parameters:
        - n_clusters (int): The number of clusters to form.
        - dist (function): The distance function to use. Defaults to Euclidean distance.
        - random_state (int): Seed for random number generator for reproducibility.
        """
        self.n_clusters = n_clusters  # Number of clusters to form
        self.dist = dist  # Distance function to compute distances
        self.rstate = np.random.RandomState(random_state)  # Random state for reproducibility (in selecting initial medoids)
        self.cluster_centers_ = []  # List to store medoids (centroid-like representatives)
        self.indices = []  # List to store indices of the current medoids


    def fit(self, X):
        """
        Computes K-Medoids clustering on the data X.

        Parameters:
        - X (numpy.ndarray): The input data, shape (n_samples, n_features)
        """
        # Shortcut to the randint method of the RandomState instance for selecting random indices
        rint = self.rstate.randint

        ## 1: Initialize medoids by randomly selecting K unique data point indices from X (μ_1,μ_2,…,μ_K)
        self.indices = [rint(X.shape[0])] # Randomly select the first medoid index
        # Select the remaining medoid indices ensuring that each selected index is unique to avoid duplicate medoids
        for _ in range(self.n_clusters - 1):
            i = rint(X.shape[0])
            while i in self.indices: # Ensure uniqueness of the selected indices
                i = rint(X.shape[0])
            self.indices.append(i)
        # Set initial cluster medoids as the data points corresponding to the selected indices
        self.cluster_centers_ = X[self.indices, :]

        ## 2. Associate each data point x^(i) (here denoted as y_pred) to the closest medoid μ_k by using in this case euclidean distance as distance metric
        self.y_pred = np.argmin(self.dist(X, self.cluster_centers_), axis=1)
        ## 3. Compute the initial Cost (given by the sum of distances of data points to their assigned medoid)
        cost = np.sum([np.sum(self.dist(X[self.y_pred == i], X[[self.indices[i]], :])) for i in set(self.y_pred)])
        new_cost = cost # 4. initialize New Cost with the initial Cost
        new_y_pred = self.y_pred.copy() # Copy of current cluster assignments
        new_indices = self.indices[:] # Copy of current medoid indices

        initial = True  # Flag to ensure at least one iteration
        ## 5. Iteratively improve medoid positions to minimize total cost
        # This means iterivately check if replacing a current medoid with another data point in the cluster reduces the total cost.
        # If this is the case confirm the swap. Repeat this until the cost no longer improves (convergence).
        while (new_cost < cost) | initial:
            initial = False # After the first iteration, initialization is done
            cost = new_cost  # Update the cost to the new_cost
            self.y_pred = new_y_pred  # Update cluster assignments
            self.indices = new_indices  # Update medoid indices
            # Let's iterate to try to find potential better medoids
            # for each cluster
            for k in range(self.n_clusters):
                # for each data point x^(i) which is not a medoid
                for r in [i for i, x in enumerate(new_y_pred == k) if x]: # here we select each data point appartaining to the given cluster k
                    if r not in self.indices: # not a medoid
                        # Swap the medoid of cluster k (μ_k) with the data point x^(i)
                        indices_temp = self.indices[:] # Create a temporary copy of current medoid indices
                        indices_temp[k] = r # Replace the k-th medoid with the current point r 
                        # Repeat step 2 and 3: based on this new set of medoids, re-assign each data point to its nearest medoid and recalculate the cost
                        # in order to get the new cost that we would obtain after the swap
                        y_pred_temp = np.argmin(self.dist(X, X[indices_temp,:]), axis=1) # Step 2
                        new_cost_temp = np.sum([np.sum(self.dist(X[y_pred_temp == i], X[[indices_temp[i]], :])) for i in set(y_pred_temp)]) # Step 3
                        # If the new cost is better (lower), update the best found so far.
                        # So we update update the cost, medoids, and assignments (basically this means confirm the swap)
                        if new_cost_temp < new_cost:
                            new_cost = new_cost_temp
                            new_indices = indices_temp
                            new_y_pred = y_pred_temp
                        # otherwise we simply don't confirm the changes (basically we undo the swap)

        # After convergence, update the medoid coordinates based on the final indices found
        self.cluster_centers_ = X[self.indices, :]


    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to (effectively clustering the data based on the final set of medoids found)

        Parameters:
        - X (numpy.ndarray): New data to predict, shape (n_samples, n_features)

        Returns:
        - y_pred (numpy.ndarray): Index of the cluster each sample belongs to, shape (n_samples,)
        """
        # Compute distances between X and the current medoids
        distances = self.dist(X, self.cluster_centers_)
        # Assign each data point to the nearest medoid
        y_pred = np.argmin(distances, axis=1)
        # return the index of the closest medoid for each point
        return y_pred
