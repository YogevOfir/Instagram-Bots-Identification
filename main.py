import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the training dataset
train_data = pd.read_csv('train.csv')

# Separate features (X_train) and target variable (y_train) for training
X_train = train_data.drop(columns=['fake'])
y_train = train_data['fake']

# Load the test dataset
test_data = pd.read_csv('test.csv')

# Separate features (X_test) and target variable (y_test) for testing
X_test = test_data.drop(columns=['fake'])
y_test = test_data['fake']

# Define the training and test data as global variables
global GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN, GLOBAL_X_TEST, GLOBAL_Y_TEST
GLOBAL_X_TRAIN = X_train
GLOBAL_Y_TRAIN = y_train
GLOBAL_X_TEST = X_test
GLOBAL_Y_TEST = y_test

# Define a list of k values and p values to try for KNN
k_values = [1, 3, 5, 7]
p_values = [1, 2, float('inf')]

# Train and evaluate KNN
print('KNN results:')
for k in k_values:
    for p in p_values:
        # Initialize the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k, p=p)

        # Train the KNN classifier on the training data
        knn.fit(GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN)

        # Calculate empirical error (training error) using mean squared error
        y_train_pred_knn = knn.predict(GLOBAL_X_TRAIN)
        emp_error_knn = mean_squared_error(GLOBAL_Y_TRAIN, y_train_pred_knn)

        # Make predictions on the test data using KNN
        y_test_pred_knn = knn.predict(GLOBAL_X_TEST)

        # Calculate true error (generalization error) using mean squared error for KNN
        true_error_knn = mean_squared_error(GLOBAL_Y_TEST, y_test_pred_knn)

        # Print results for KNN with explicit k and p values
        print(f'KNN with k={k} and LP norm = {p}: empirical error: {emp_error_knn:.2f}, true error on test data: {true_error_knn:.2f}')

# Define a list of C values to try for SVM
C_values = [0.1, 1, 10]

# Train and evaluate SVM
print('SVM results:')
for C in C_values:
    # Initialize the SVM classifier
    svm = SVC(C=C, kernel='linear')  # Use linear kernel for simplicity

    # Train the SVM classifier on the training data
    svm.fit(GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN)

    # Calculate empirical error (training error) using mean squared error
    y_train_pred_svm = svm.predict(GLOBAL_X_TRAIN)
    emp_error_svm = mean_squared_error(GLOBAL_Y_TRAIN, y_train_pred_svm)

    # Make predictions on the test data using SVM
    y_test_pred_svm = svm.predict(GLOBAL_X_TEST)

    # Calculate true error (generalization error) using mean squared error for SVM
    true_error_svm = mean_squared_error(GLOBAL_Y_TEST, y_test_pred_svm)

    # Print results for SVM with explicit C value
    print(f'SVM with C={C}: empirical error: {emp_error_svm:.2f}, true error on test data: {true_error_svm:.2f}')

# Define a list of n_estimators values to try for AdaBoost
n_estimators_values = [1, 100, 1000]

# Train and evaluate AdaBoost with SVM as base estimator
print('AdaBoost with SVM results:')
for n_estimators in n_estimators_values:
    # Initialize AdaBoost classifier with SVM as base estimator
    adaboost = AdaBoostClassifier(SVC(C=1, kernel='linear'), n_estimators=n_estimators, algorithm='SAMME', random_state=42)

    # Train AdaBoost classifier on the training data
    adaboost.fit(GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN)

    # Calculate empirical error (training error) using mean squared error
    y_train_pred_adaboost = adaboost.predict(GLOBAL_X_TRAIN)
    emp_error_adaboost = mean_squared_error(GLOBAL_Y_TRAIN, y_train_pred_adaboost)

    # Make predictions on the test data using AdaBoost
    y_test_pred_adaboost = adaboost.predict(GLOBAL_X_TEST)

    # Calculate true error (generalization error) using mean squared error for AdaBoost
    true_error_adaboost = mean_squared_error(GLOBAL_Y_TEST, y_test_pred_adaboost)

    # Print results for AdaBoost with explicit n_estimators value
    print(f'AdaBoost with n_estimators={n_estimators}: empirical error: {emp_error_adaboost:.2f}, true error on test data: {true_error_adaboost:.2f}')

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(GLOBAL_X_TRAIN)
X_test_scaled = scaler.transform(GLOBAL_X_TEST)

# Define a list of regularization parameter values (C) and solvers to try for logistic regression
C_values_lr = [0.1, 1, 10]
solvers = ['lbfgs', 'liblinear', 'saga']

# Train and evaluate Logistic Regression with different C values and solvers
print('Logistic Regression results:')
for C_lr in C_values_lr:
    for solver in solvers:
        # Initialize the Logistic Regression classifier
        logreg = LogisticRegression(C=C_lr, solver=solver, max_iter=1000)

        # Train the Logistic Regression classifier on the scaled training data
        logreg.fit(X_train_scaled, GLOBAL_Y_TRAIN)

        # Make predictions on the scaled test data using Logistic Regression
        y_test_pred_lr = logreg.predict(X_test_scaled)

        # Calculate true error (generalization error) using mean squared error for Logistic Regression
        true_error_lr = mean_squared_error(GLOBAL_Y_TEST, y_test_pred_lr)

        # Print results for Logistic Regression with explicit C and solver
        print(f'Logistic Regression with C={C_lr} and solver={solver}: true error on test data: {true_error_lr:.2f}')
