import numpy as np

class NaiveBayes:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.attack_cat = None
        self.prior = {}
        self.mean = {}
        self.variance = {}

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        self.attack_cat = np.unique(y)

        # hitung probabilitas prior
        self.prior = {cat: np.sum(y == cat) / len(y) for cat in self.attack_cat}
        
        # hitung mean dan variance untuk setiap fitur dan kategori
        for cat in self.attack_cat:
            indices = np.where(y == cat)[0]
            self.mean[cat] = np.mean(self.X_train[indices], axis=0)
            self.variance[cat] = np.var(self.X_train[indices], axis=0)

        return self

    # Gaussian Probability Density Function.
    def gaussian_pdf(self, x, mean, var):
        eps = 1e-9  # smoothing
        coeff = 1.0 / np.sqrt(2.0 * np.pi * (var + eps))
        exponent = -((x - mean) ** 2) / (2.0 * (var + eps))
        return coeff * np.exp(exponent)

    def predict(self, X):
        X = np.array(X)
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Number of features in X does not match the number of features in X_train")
        
        y_pred = []
        for x in X:
            posterior = {}
            for cat in self.attack_cat:
                posterior[cat] = np.log(self.prior[cat])
                for i in range(self.X_train.shape[1]):
                    posterior[cat] += np.log(self.gaussian_pdf(x[i], self.mean[cat][i], self.variance[cat][i]))
            y_pred.append(max(posterior, key=posterior.get))
        
        return np.array(y_pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)