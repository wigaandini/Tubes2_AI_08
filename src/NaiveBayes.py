import numpy as np

class NaiveBayes:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.attack_cat = None
        self.prior = {}
        self.likelihood = {}
    
    # X harus udah dipreprocess dulu untuk convert numeric ke categorical
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        self.attack_cat = np.unique(y)

        # hitung probabilitas prior
        self.prior = {cat: np.sum(y == cat) / len(y) for cat in self.attack_cat}
        
        # hitung probabilitas likelihood
        # untuk setiap fitur
        for i in range(self.X_train.shape[1]):
            self.likelihood[i] = {}
            # untuk setiap kategori
            for cat in self.attack_cat:
                self.likelihood[i][cat] = {}
                indices = np.where(y == cat)[0]
                # hitung probabilitas nilai fitur i diberikan kategori cat
                for value in np.unique(self.X_train[:, i]):
                    # Laplace smoothing
                    # numerator ditambah 1, denominator ditambah jumlah nilai unik di fitur i
                    self.likelihood[i][cat][value] = (
                        np.sum(self.X_train[indices, i] == value) + 1) / (len(indices) + len(np.unique(self.X_train[:, i])))

        return self
    
    def predict(self, X):
        X = np.array(X)
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Number of features in X does not match the number of features in X_train")
        
        y_pred = []
        for x in X:
            posterior = {}
            for cat in self.attack_cat:
                # work in the log probability space
                # When you multiply one small number by another small number, you get a very small number.
                posterior[cat] = np.log(self.prior[cat])
                for i in range(self.X_train.shape[1]):
                    posterior[cat] += np.log(self.likelihood[i][cat].get(x[i], 1 / (len(self.y_train[self.y_train == cat]) + len(np.unique(self.X_train[:, i])))))
            y_pred.append(max(posterior, key=posterior.get))
        
        return np.array(y_pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)