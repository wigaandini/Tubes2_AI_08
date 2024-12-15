from sklearn.naive_bayes import MultinomialNB, GaussianNB, CategoricalNB

class NaiveBayesSKlearn:
    def __init__(self, model_type='categorical'):
        if model_type == 'multinomial':
            self.model = MultinomialNB()
        elif model_type == 'gaussian':
            self.model = GaussianNB()
        elif model_type == 'categorical':
            self.model = CategoricalNB()
        else:
            raise ValueError("model_type harus salah satu dari 'multinomial', 'gaussian', atau 'categorical'")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)