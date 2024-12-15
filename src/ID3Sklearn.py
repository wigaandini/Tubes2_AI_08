import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import math

class ID3Sklearn:
    def __init__(self, criterion='entropy', max_depth=3, min_samples_split=10, random_state=42):
        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def _scaler_fitTransform(self, X):
        return self.scaler.fit_transform(X)
    
    def _scaler_transform(self, X):
        return self.scaler.transform(X)
    
    def _scaler_fitTransform(self, Y):
        return self.scaler.fit_transform(Y)
    
    def _scaler_transform(self, Y):
        return self.scaler.transform(Y)

    def fit(self, X, y):
        X = self._scaler_fitTransform(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self._scaler_transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = self._transform(X)
        return self.model.predict_proba(X)

    def performance_report(self, X, y):
        print("Accuracy:", log_loss(y, self.predict_proba(X)))
        print("\nClassification Report:")
        print(classification_report(y, self.predict(X)))

