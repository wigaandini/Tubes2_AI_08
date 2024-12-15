

<h1 align="center"> Tugas Besar 2 Intelegensi Artifisial </h1>
<h1 align="center">  Implementasi Algoritma Pembelajaran Mesin </h1>

## Table of Contents
1. [General Information](#general-information)
2. [Contributors](#contributors)
3. [Features](#features)
4. [Requirements Program](#required_program)
5. [How to Run The Program](#how-to-run-the-program)


## General Information
This project focuses on the implementation of machine learning algorithms to classify different types of cyber attacks in the UNSW-NB15 dataset. The dataset consists of network traffic data containing various types of activities, both normal and associated with cyber attacks. It includes 10 activity categories: 9 attack types (Fuzzers, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) and 1 normal activity. The target variable `attack_cat` represents the attack category.

The task requires implementing three machine learning algorithms from scratch: **K-Nearest Neighbors (KNN)**, **Gaussian Naive Bayes**, and **ID3**. The same algorithms are then applied using the `scikit-learn` library to compare results between the manual implementation and the standard library implementation. This project also involves various stages of data preprocessing, feature selection, and model evaluation.


## Contributors
### **Kelompok 8**
|   NIM    |                  Nama                  |
| :------: | :------------------------------------: |
| 13522018 |           Ibrahim Ihsan Rasyid         |
| 13522053 |       Erdianti Wiga Putri Andini       |
| 13522097 |    Ellijah Darrellshane Suryanegara    |
| 13522114 |      Muhammad Dava Fathurrahman        |


## Features
Features that used in this program are:
### **K-Nearest Neighbors (KNN)**

KNN is a simple and effective classification algorithm based on proximity. It classifies data points by looking at the `k` closest data points (neighbors) and choosing the majority class among them. 

**Key Points:**
- **Distance Metrics**: The distance between data points can be calculated using various metrics, such as **Euclidean**, **Manhattan**, or **Minkowski** distance.
- **Parameter**: `k` determines the number of neighbors considered for classifying a data point. A smaller `k` can be more sensitive to noise, while a larger `k` provides a smoother decision boundary.
- **Strength**: The model works well for datasets where similar instances tend to have similar labels. It is easy to implement and does not make strong assumptions about the data.

**Implementation**:
- The KNN algorithm is implemented from scratch, where it calculates the distance between data points and selects the most frequent class among the closest neighbors.

### **Gaussian Naive Bayes**

Naive Bayes is a probabilistic classifier based on Bayes' theorem, which assumes that all features are independent (naive assumption). It calculates the probability of each class label given the input features and selects the class with the highest probability.

**Key Points:**
- **Gaussian Assumption**: The Naive Bayes classifier assumes that the features follow a normal (Gaussian) distribution. This assumption simplifies the calculation of likelihoods for continuous data.
- **Probabilistic Approach**: Given input features, it computes the posterior probability for each class using prior probabilities and likelihoods, then classifies the input to the class with the highest posterior.
- **Strength**: It works well with high-dimensional data, and it's fast, making it a great choice for initial classification models.

**Implementation**:
- The Gaussian Naive Bayes algorithm is implemented from scratch by calculating the likelihood of each feature under a Gaussian distribution and combining them to compute class probabilities.

### **ID3 (Iterative Dichotomiser 3)**

ID3 is a decision tree algorithm used for classification that builds a tree by splitting the data based on the feature that provides the most information gain. It uses the concept of entropy from information theory to make decisions.

**Key Points:**
- **Entropy**: A measure of impurity or uncertainty in a dataset. ID3 selects the feature that reduces the entropy the most (i.e., maximizes information gain) at each step.
- **Recursive Tree Building**: The algorithm recursively selects the best feature to split the data into subsets, and this process continues until the dataset is homogenous or certain stopping conditions are met.
- **Strength**: ID3 is interpretable (i.e., easy to understand), as the resulting model is a decision tree. It's very effective for categorical data and can be extended to handle continuous features by discretizing them.

**Implementation**:
- The ID3 algorithm is implemented from scratch, where the tree is built by calculating entropy and information gain for each feature and selecting the best feature to split the data recursively.


## Requirements Program
|   NO   |  Required Program    |
| :----: | -------------------- |
|   1    | `numpy`              |                            
|   2    | `pandas`             |
|   3    | `scikit-learn`       |
|   4    | `matplotlib`         |

Install the dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## How to Run The Program
1. Open your terminal.
2. Clone this repository by typing `git clone https://github.com/wigaandini/Tubes2_AI_08.git` in the terminal.
3. Change the directory using `cd src`.
4. Open the file `AI_Tubes2_Kelompok 8.ipynb` or you can use jupyter notebook by typing `jupyter notebook` in terminal then open the file `AI_Tubes2_Kelompok 8.ipynb`.
5. Select `Run All` on the top of program (click the Runtime button first).
6. Look at the cell that calculate accuracy kNN model, Naive Bayes model, and ID3 model.