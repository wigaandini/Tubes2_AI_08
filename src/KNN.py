import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, metric='euclidean', p=2): # default value
        self.k = k
        self.metric = metric.lower()
        self.p = p
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1, x2):
        return np.power(np.sum(np.power(np.abs(x1 - x2), self.p)), 1/self.p)
    
    def _calculate_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError("Please choose a valid distance metric: 'euclidean', 'manhattan', or 'minkowski'")
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict(self, X):
        X = np.array(X)
        y_pred = []
        
        for x in X:
            # hitung jarak antara x dan semua data train
            distances = []
            for x_train in self.X_train:
                dist = self._calculate_distance(x, x_train)
                distances.append(dist)
            
            # cari k index terdekat
            k_indices = np.argsort(distances)[:self.k]
            
            # ambil label dari k index terdekat
            k_nearest_labels = self.y_train[k_indices]
            
            # label yang paling sering muncul
            most_common = Counter(k_nearest_labels).most_common(1)
            y_pred.append(most_common[0][0])
        
        return np.array(y_pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def predict_proba(self, X):
        X = np.array(X)
        probabilities = []
        
        for x in X:
            # hitung jarak antara x dan semua data train
            distances = []
            for x_train in self.X_train:
                dist = self._calculate_distance(x, x_train)
                distances.append(dist)
            
            # cari k index terdekat
            k_indices = np.argsort(distances)[:self.k]
            
            # ambil label dari k index terdekat
            k_nearest_labels = self.y_train[k_indices]
            
            # hitung probabilitas masing-masing kelas
            class_counts = Counter(k_nearest_labels)
            proba = []
            for class_label in self.classes_:
                class_prob = class_counts.get(class_label, 0) / self.k
                proba.append(class_prob)
            
            probabilities.append(proba)
        
        return np.array(probabilities)