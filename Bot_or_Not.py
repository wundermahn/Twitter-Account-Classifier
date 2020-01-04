from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import export_graphviz
from subprocess import call
import pandas as pd, numpy as np, warnings, pickle

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

# Load data
data = pd.read_csv("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Project\\Code\\Final_DF.csv")
print(data)
exit()
labels = list(data['class'])
data = data.drop('class', axis=1)
data = data.drop('username', axis=1)

# Function to test the data
def test_reg_data(data, label):

    precision_list = []
    recall_list = []
    f1_list = []


    for x in range(10):
        # Train / test split the data
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.33, shuffle=True)  

        # Create the classifier
        rf = RandomForestClassifier(bootstrap=True, max_depth=100, max_features='log2', min_samples_leaf=2, min_impurity_split=2, n_estimators=100)

        # Fit the model
        rf.fit(X_train, y_train)

        # Collect Predictions
        rf_preds = rf.predict(X_test)

        # Collect metrics
        rf_precision, rf_recall, rf_f1 = collect_metrics(rf_preds, y_test)
        precision_list.append(rf_precision)
        recall_list.append(rf_recall)
        f1_list.append(rf_f1)

    rf_precision = np.average(precision_list)
    rf_recall = np.average(recall_list)
    np.average(f1_list)

    for name, importance in zip(X_test.columns, rf.feature_importances_):
        print(name, "=", importance)

    # Return them
    return rf_precision, rf_recall, rf_f1, rf

print("Kicking off training...")

# Get the best results from gridsearch cv
rf_precision, rf_recall, rf_f1, rf= test_reg_data(data, labels)

# Print out the results
print("             RESULTS             ")
print("############################################")
print("RF      | Recall: {} | Precision: {} | F1: {}".format(rf_recall, rf_precision, rf_f1))
print("############################################")

estimator = rf.estimators_[5]

export_graphviz(estimator, 
                out_file='C:\\Users\\Kelly\\Desktop\\tree.dot', 
                feature_names = ['count_bot', 'count_human', '%_bot_tweets',  'following',  'followers'],
                class_names = ['human', 'bot'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
#call(['dot', '-Tpng', 'C:\\Users\\Kelly\\Desktop\\tree.dot', '-o', 'C:\\Users\\Kelly\\Desktop\\tree.png', '-Gdpi=600'])                
pickle.dump(rf, open("C:\\Users\\Kelly\\Desktop\\rf.pkl",'wb'))