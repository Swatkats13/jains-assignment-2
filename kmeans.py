

import numpy as np

class KMeans:
    def __init__(self, n_clusters, init_method='random', max_iters=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.assignment = None  # Tracks cluster assignment of points

    def initialize_centroids(self, X, manual_centroids=None):
        if self.init_method == 'random':
            return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init_method == 'farthest_first':
            return self.farthest_first_initialization(X)
        elif self.init_method == 'kmeans++':
            return self.kmeans_plus_plus_initialization(X)
        elif self.init_method == 'manual':
            # Use the manually provided centroids from user clicks
            return np.array(manual_centroids)

    def farthest_first_initialization(self, X):
        centroids = [X[np.random.choice(len(X))]]  # Start with a random point
        for _ in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - centroid) for centroid in centroids]) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self, X):
        centroids = [X[np.random.choice(len(X))]]  # Start with a random point
        for _ in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - centroid) for centroid in centroids]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            next_centroid = X[np.searchsorted(cumulative_probabilities, r)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def fit(self, X, manual_centroids=None):
        self.centroids = self.initialize_centroids(X, manual_centroids=manual_centroids)
        for _ in range(self.max_iters):  # Max 100 iterations to ensure convergence
            clusters = self.assign_clusters(X)
            new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(new_centroids, self.centroids, rtol=self.tol):
                break
            self.centroids = new_centroids

    def step(self, X):
        clusters = self.assign_clusters(X)
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.n_clusters)])
        self.centroids = new_centroids

    def assign_clusters(self, X):
        distances = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in X])
        self.assignment = np.argmin(distances, axis=1)
        return self.assignment

    def predict(self, X):
        return self.assign_clusters(X)

