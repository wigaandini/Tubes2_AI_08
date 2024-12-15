import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import math


class ID3Sklearn:
    def __init__(self):
        self.entropy_outcome = None  # Entropy of the target column
        self.feature_dict = {}

    def calculate_entropy(self, data, target_column):
        total_rows = len(data)
        target_values = data[target_column].value_counts()

        entropy = 0
        for count in target_values:
            proportion = count / total_rows
            entropy -= proportion * math.log2(proportion) if proportion > 0 else 0

        return entropy

    def calculate_information_gain(self, data, feature, target_column):
        total_entropy = self.calculate_entropy(data, target_column)
        unique_values = data[feature].unique()

        weighted_entropy = 0
        for value in unique_values:
            subset = data[data[feature] == value]
            proportion = len(subset) / len(data)
            weighted_entropy += proportion * self.calculate_entropy(subset, target_column)

        information_gain = total_entropy - weighted_entropy
        return information_gain

    def assess_best_feature(self, data):
        self.feature_dict = {}
        target_column = "attack_cat"

        for column in data.columns[:-1]:  # Exclude the target column
            information_gain = self.calculate_information_gain(data, column, target_column)
            self.feature_dict[column] = information_gain

        # Find the feature with the maximum information gain
        best_feature = max(self.feature_dict, key=self.feature_dict.get)
        return best_feature

    def createTree(self, data, max_depth=3):
        selected_feature = self.assess_best_feature(data)
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        
        X = data[[selected_feature]]
        y = data.iloc[:, -1]  # Target column

        clf.fit(X, y)

        plt.figure(figsize=(8, 6))
        plot_tree(
            clf, 
            feature_names=[selected_feature], 
            class_names=[str(cls) for cls in y.unique()], 
            filled=True, 
            rounded=True
        )
        plt.show()

    def id3(self, data, target_column, features):
        # If all target values are the same, return the value
        if len(data[target_column].unique()) == 1:
            return data[target_column].iloc[0]

        # If no features are left, return the majority class
        if len(features) == 0:
            return data[target_column].mode().iloc[0]

        # Select the best feature
        best_feature = max(features, key=lambda x: self.calculate_information_gain(data, x, target_column))

        # Initialize the tree with the best feature
        tree = {best_feature: {}}

        # Remove the best feature from the list
        remaining_features = [f for f in features if f != best_feature]

        # Iterate through unique values of the best feature
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            if subset.empty:
                # Assign the majority class if the subset is empty
                tree[best_feature][value] = data[target_column].mode().iloc[0]
            else:
                # Recursive call
                tree[best_feature][value] = self.id3(subset, target_column, remaining_features)

        return tree

