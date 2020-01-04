import pandas as pd, numpy as np, re, string, time, gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# FOR TESTING ONLY
import warnings
warnings.filterwarnings("ignore")

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

def minmaxscale(data):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return df_scaled

# Create Classifiers
knn = KNeighborsClassifier(n_neighbors = 3)
rf = RandomForestClassifier(n_estimators = 100)
mlp = MLPClassifier((100,))
svm = SVC(gamma='auto')

# Load data
data = pd.read_csv("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Project\\Code\\Final_DF.csv")
labels = list(data['class'])
data = data.drop('class', axis=1)
data = data.drop('username', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size = 0.2)

# Train Models
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
mlp.fit(minmaxscale(X_train), y_train)
svm.fit(X_train, y_train)

# Predict data with models
knn_preds = knn.predict(X_test)
rf_preds = rf.predict(X_test)
mlp_preds = mlp.predict(minmaxscale(X_test))
svm_preds = svm.predict(X_test)

# Collect metrics
knn_precision, knn_recall, knn_f1 = collect_metrics(knn_preds, y_test)
rf_precision, rf_recall, rf_f1 = collect_metrics(rf_preds, y_test)
mlp_precision, mlp_recall, mlp_f1 = collect_metrics(mlp_preds, y_test)
svm_precision, svm_recall, svm_f1 = collect_metrics(svm_preds, y_test)

# Pretty print the results
print("KNN     | Recall: {} | Precision: {} | F1: {}".format(knn_recall, knn_precision, knn_f1))
print("RF      | Recall: {} | Precision: {} | F1: {}".format(rf_recall, rf_precision, rf_f1))
print("MLP     | Recall: {} | Precision: {} | F1: {}".format(mlp_recall, mlp_precision, mlp_f1))
print("SVM     | Recall: {} | Precision: {} | F1: {}".format(svm_recall, svm_precision, svm_f1))