import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

class ID3:
    def __init__(self):
        self.tree = {}

    def calc_total_entropy(self, train_data, label, class_list):
        total_row = train_data.shape[0]
        total_entr = 0
        
        for c in class_list:
            total_class_count = train_data[train_data[label] == c].shape[0]
            if total_class_count != 0:
                total_class_entr = - (total_class_count / total_row) * np.log2(total_class_count / total_row)
                total_entr += total_class_entr
        
        return total_entr

    def calc_entropy(self, feature_value_data, label, class_list):
        class_count = feature_value_data.shape[0]
        entropy = 0
        
        for c in class_list:
            label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
            if label_class_count != 0:
                probability_class = label_class_count / class_count
                entropy_class = - probability_class * np.log2(probability_class)
                entropy += entropy_class
        
        return entropy

    def calc_info_gain(self, feature_name, train_data, label, class_list):
        feature_value_list = train_data[feature_name].unique()
        total_row = train_data.shape[0]
        feature_info = 0.0

        for feature_value in feature_value_list:
            feature_value_data = train_data[train_data[feature_name] == feature_value]
            feature_value_count = feature_value_data.shape[0]
            feature_value_entropy = self.calc_entropy(feature_value_data, label, class_list)
            feature_value_probability = feature_value_count / total_row
            feature_info += feature_value_probability * feature_value_entropy

        return self.calc_total_entropy(train_data, label, class_list) - feature_info

    def find_most_informative_feature(self, train_data, label, class_list):
        feature_list = train_data.columns.drop(label)
        max_info_gain = -1
        max_info_feature = None

        for feature in feature_list:
            feature_info_gain = self.calc_info_gain(feature, train_data, label, class_list)
            if max_info_gain < feature_info_gain:
                max_info_gain = feature_info_gain
                max_info_feature = feature

        return max_info_feature

    def generate_sub_tree(self, feature_name, train_data, label, class_list):
        feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
        tree = {}

        for feature_value, count in feature_value_count_dict.items():
            feature_value_data = train_data[train_data[feature_name] == feature_value]

            assigned_to_node = False
            for c in class_list:
                class_count = feature_value_data[feature_value_data[label] == c].shape[0]

                if class_count == count:
                    tree[feature_value] = c
                    train_data = train_data[train_data[feature_name] != feature_value]
                    assigned_to_node = True
            
            if not assigned_to_node:
                tree[feature_value] = "?"

        return tree, train_data

    def make_tree(self, root, prev_feature_value, train_data, label, class_list):
        if train_data.shape[0] != 0:
            max_info_feature = self.find_most_informative_feature(train_data, label, class_list)
            tree, train_data = self.generate_sub_tree(max_info_feature, train_data, label, class_list)
            next_root = None

            if prev_feature_value is not None:
                root[prev_feature_value] = {}
                root[prev_feature_value][max_info_feature] = tree
                next_root = root[prev_feature_value][max_info_feature]
            else:
                root[max_info_feature] = tree
                next_root = root[max_info_feature]

            for node, branch in list(next_root.items()):
                if branch == "?":
                    feature_value_data = train_data[train_data[max_info_feature] == node]
                    self.make_tree(next_root, node, feature_value_data, label, class_list)

    def fit(self, train_data, label):
        self.tree = {}
        class_list = train_data[label].unique()
        self.make_tree(self.tree, None, train_data, label, class_list)
        return self.tree

    def predict(self, instance):
        def _predict(tree, instance):
            if not isinstance(tree, dict):
                return tree
            else:
                root_node = next(iter(tree))
                feature_value = instance[root_node]
                if feature_value in tree[root_node]:
                    return _predict(tree[root_node][feature_value], instance)
                else:
                    return None

        return _predict(self.tree, instance)
    

from sklearn.preprocessing import LabelEncoder

label_column = 'attack_cat'  # Specify the column name of the labels

# Combine the features and labels into a single DataFrame
train_data = X_train_id3.copy()  # Make a copy of the features
train_data[label_column] = y_train_essentials.reset_index(drop=True)  # Add the labels as a new colum

print(train_data)
id3 = ID3()
tree = id3.fit(train_data, label_column)

# Display the tree
print("Generated Decision Tree:")
print(tree)

test_data = X_val_id3.copy()  # Make a copy of the features
test_data[label_column] = y_val_essentials.reset_index(drop=True)  # Add the labels as a new column

# Predict for each instance in the test set
predictions = test_data.apply(lambda x: id3.predict(x), axis=1)

# Evaluate predictions
print("\nAccuracy:", accuracy_score(test_data[label_column], predictions))
print("\nClassification Report:\n", classification_report(test_data[label_column], predictions))
