from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import numpy as np
import time

# Function to collect required F1, Precision, and Recall Metrics
def collect_metrics(actuals, preds):
    # Create a confusion matrix
    matr = confusion_matrix(actuals, preds, labels=[0, 1])
    # Retrieve TN, FP, FN, and TP from the matrix
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(actuals, preds).ravel()

    # Compute precision
    precision = true_positive / (true_positive + false_positive)
    # Compute recall
    recall = true_positive / (true_positive + false_negative)
    # Compute F1
    f1 = 2*((precision*recall)/(precision + recall))

    # Return results
    return precision, recall, f1

# Load data
data = pd.read_csv("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Project\\Code\\Final_DF.csv")
labels = list(data['class'])
data = data.drop('class', axis=1)
data = data.drop('username', axis=1)

# Set the parameter grid to sweep
param_grid = {
    'bootstrap': [True],
    'max_depth': [None, 25, 100],
    'max_features': ['sqrt', 'log2', 'auto'],
    'min_samples_leaf': [2, 3, 5],
    'min_samples_split': [2, 3, 5],
    'n_estimators': [100, 1000, 5000, 10000]
}# RF Param Grid

# Function to test the data
def test_reg_data(params, data, label):
    # Train / test split the data
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2)  

    # Create the classifier
    rf = RandomForestClassifier()
    # Get GridSearch ready
    grid_search = GridSearchCV(estimator = rf, param_grid = params, cv = 3, n_jobs = -1, verbose = 2, scoring='f1')
    # Fit it on the data
    grid_search.fit(X_train, y_train)

    # Print out the best parameters
    reg_best_params = grid_search.best_params_    
    reg_best_score = grid_search.best_score_
    rf_preds = grid_search.predict(X_test)
    rf_precision, rf_recall, rf_f1 = collect_metrics(rf_preds, y_test)

    return reg_best_params, reg_best_score, rf_precision, rf_recall, rf_f1

print("Kicking off training...")

# Get the best results from gridsearch cv
reg_best_params, reg_best_score, rf_precision, rf_recall, rf_f1 = test_reg_data(param_grid, data, labels)

# Print out the results
print("############################################")
print("             RESULTS             ")
print('Regular Data Best Parameters: {}'.format(reg_best_params))
print('Regular Data Best Score: {}'.format(reg_best_score))
print("RF      | Recall: {} | Precision: {} | F1: {}".format(rf_recall, rf_precision, rf_f1))
print("############################################")
print("             RESULTS             ")
