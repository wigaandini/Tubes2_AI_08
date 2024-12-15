import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, metric='euclidean', p=2, batch_size=1000):
        self.k = k
        self.metric = metric.lower()
        self.p = p  # Added parameter p for Minkowski distance
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def _compute_distances_batch(self, X_batch):
        if self.metric == 'euclidean':
            test_norm = np.sum(X_batch**2, axis=1)[:, np.newaxis]
            train_norm = np.sum(self.X_train**2, axis=1)

            distances = np.zeros((X_batch.shape[0], self.X_train.shape[0]), dtype=np.float32)
            chunk_size = 1000 

            for i in range(0, self.X_train.shape[0], chunk_size):
                end_idx = min(i + chunk_size, self.X_train.shape[0])
                distances[:, i:end_idx] = -2 * np.dot(X_batch, self.X_train[i:end_idx].T)

            distances += test_norm + train_norm
            return np.sqrt(np.maximum(distances, 0))

        elif self.metric == 'manhattan':
            distances = np.zeros((X_batch.shape[0], self.X_train.shape[0]), dtype=np.float32)
            chunk_size = 1000

            for i in range(0, self.X_train.shape[0], chunk_size):
                end_idx = min(i + chunk_size, self.X_train.shape[0])
                distances[:, i:end_idx] = np.sum(
                    np.abs(X_batch[:, np.newaxis] - self.X_train[i:end_idx]),
                    axis=2
                )
            return distances
            
        elif self.metric == 'minkowski':
            distances = np.zeros((X_batch.shape[0], self.X_train.shape[0]), dtype=np.float32)
            chunk_size = 1000

            for i in range(0, self.X_train.shape[0], chunk_size):
                end_idx = min(i + chunk_size, self.X_train.shape[0])
                # Calculate difference matrix
                diff = X_batch[:, np.newaxis] - self.X_train[i:end_idx]
                # Apply p-norm formula
                distances[:, i:end_idx] = np.sum(
                    np.abs(diff) ** self.p,
                    axis=2
                ) ** (1/self.p)
            return distances
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}. Choose from 'euclidean', 'manhattan', or 'minkowski'")

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        predictions = []

        for i in range(0, X.shape[0], self.batch_size):
            batch_end = min(i + self.batch_size, X.shape[0])
            X_batch = X[i:batch_end]

            distances = self._compute_distances_batch(X_batch)

            k_nearest_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]

            batch_predictions = []
            for labels in k_nearest_labels:
                batch_predictions.append(Counter(labels).most_common(1)[0][0])

            predictions.extend(batch_predictions)

        return np.array(predictions)

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float32)
        all_probabilities = []

        for i in range(0, X.shape[0], self.batch_size):
            batch_end = min(i + self.batch_size, X.shape[0])
            X_batch = X[i:batch_end]

            distances = self._compute_distances_batch(X_batch)
            k_nearest_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]

            batch_probabilities = []
            for labels in k_nearest_labels:
                counts = Counter(labels)
                probs = [counts.get(label, 0) / self.k for label in self.classes_]
                batch_probabilities.append(probs)

            all_probabilities.extend(batch_probabilities)

        return np.array(all_probabilities)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)