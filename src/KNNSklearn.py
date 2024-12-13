from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNSklearn:
    def __init__(self, k=3, metric='euclidean', p=2): # default value
        self.k = k
        self.metric = metric.lower()
        self.p = p
        
        metric_mapping = {
            'euclidean': 'euclidean',
            'manhattan': 'manhattan',
            'minkowski': 'minkowski'
        }
        
        if self.metric not in metric_mapping:
            raise ValueError("Please choose a valid distance metric: 'euclidean', 'manhattan', or 'minkowski'")
            
        # inisialisasi model KNN
        self.classifier = KNeighborsClassifier(
            n_neighbors=self.k,
            metric=metric_mapping[self.metric],
            p=self.p
        )
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classifier.fit(X, y)
        return self
    
    def predict(self, X):
        X = np.array(X)
        return self.classifier.predict(X)
    
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        return self.classifier.score(X, y)
    
    def predict_proba(self, X):
        X = np.array(X)
        return self.classifier.predict_proba(X)