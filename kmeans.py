import numpy as np

class KMeans:
    def __init__(self, n_clusters, init_method='random', max_iters=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.assignment = None
    
    def initialize_centroids(self, X):
        if self.init_method == 'random':
            return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init_method == 'farthest_first':
            return self.farthest_first_initialization(X)
        elif self.init_method == 'kmeans++':
            return self.kmeans_plus_plus_initialization(X)

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
        if manual_centroids is not None:
            self.centroids = np.array(manual_centroids)
        else:
            self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X)
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = X[clusters == i]
                if len(cluster_points) == 0:
                    new_centroids.append(self.centroids[i])  # Handle empty cluster
                else:
                    new_centroids.append(cluster_points.mean(axis=0))

            new_centroids = np.array(new_centroids)

            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break  # Converged
            self.centroids = new_centroids

        self.assignment = clusters

    def step(self, X):
        # Perform one iteration of KMeans (stepping through the algorithm)
        clusters = self.assign_clusters(X)
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[clusters == i]
            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[i])  # Handle empty cluster
            else:
                new_centroids.append(cluster_points.mean(axis=0))

        self.centroids = np.array(new_centroids)
        self.assignment = clusters

    def assign_clusters(self, X):
        distances = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self.assign_clusters(X)
