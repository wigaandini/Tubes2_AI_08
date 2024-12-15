import numpy as np
from collections import Counter

class ID3:
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None):
        """
        Node structure for the ID3 decision tree.
        """
        self.feature = feature  # Feature index used for splitting
        self.value = value      # Value used for splitting
        self.results = results  # Final classification (leaf node)
        self.true_branch = true_branch  # Branch for values <= value
        self.false_branch = false_branch  # Branch for values > value

    @staticmethod
    def entropy(data):
        """
        Calculate entropy of a dataset.
        """
        counts = np.bincount(data)
        probabilities = counts / len(data)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    @staticmethod
    def split_data(X, y, feature, value):
        """
        Split the dataset based on a feature and a value.
        """
        true_indices = np.where(X[:, feature] <= value)[0]
        false_indices = np.where(X[:, feature] > value)[0]
        true_X, true_y = X[true_indices], y[true_indices]
        false_X, false_y = X[false_indices], y[false_indices]
        return true_X, true_y, false_X, false_y

    def build_tree(self, X, y):
        """
        Build the decision tree recursively.
        """
        # If all labels are the same, return a leaf node
        if len(set(y)) == 1:
            return ID3(results=Counter(y).most_common(1)[0][0])

        # Initialize variables to track the best split
        best_gain = 0
        best_criteria = None
        best_sets = None
        n_features = X.shape[1]
        current_entropy = self.entropy(y)

        for feature in range(n_features):
            # Iterate through unique values of the feature
            feature_values = np.unique(X[:, feature])
            for value in feature_values:
                true_X, true_y, false_X, false_y = self.split_data(X, y, feature, value)
                if len(true_y) == 0 or len(false_y) == 0:  # Skip invalid splits
                    continue

                # Calculate information gain
                true_entropy = self.entropy(true_y)
                false_entropy = self.entropy(false_y)
                p = len(true_y) / len(y)
                gain = current_entropy - p * true_entropy - (1 - p) * false_entropy

                # Update the best criteria if the gain is higher
                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (feature, value)
                    best_sets = (true_X, true_y, false_X, false_y)

        # If a valid split is found, create branches recursively
        if best_gain > 0:
            true_branch = self.build_tree(best_sets[0], best_sets[1])
            false_branch = self.build_tree(best_sets[2], best_sets[3])
            return ID3(feature=best_criteria[0], value=best_criteria[1],
                       true_branch=true_branch, false_branch=false_branch)

        # If no split improves information gain, return a leaf node
        return ID3(results=Counter(y).most_common(1)[0][0])

    def predict(self, X):
        """
        Predict the class for each sample in the dataset.
        """
        results = []
        for sample in X:
            results.append(self._classify(sample))
        return np.array(results)

    def _classify(self, sample):
        """
        Recursively classify a single sample.
        """
        if self.results is not None:  # If it's a leaf node, return the result
            return self.results
        if sample[self.feature] <= self.value:
            return self.true_branch._classify(sample)
        else:
            return self.false_branch._classify(sample)