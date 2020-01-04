import pandas as pd, numpy as np, re, string, time, gc, warnings, sys, pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

warnings.filterwarnings("ignore")

# List of NLTK stop words
stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 
'again', 'there', 'about', 'once', 'during', 'out', 'very', 
'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 
'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 
'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 
'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 
'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 
'were', 'her', 'more', 'himself', 'this', 'down', 'should', 
'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 
'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 
'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 
'does', 'yourselves', 'then', 'that', 'because', 'what', 
'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 
'he', 'you', 'herself', 'has', 'just', 'where', 'too', 
'only', 'myself', 'which', 'those', 'i', 'after', 'few', 
'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 
'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

def print_slow(a, b):
    for letter in b:
        sys.stdout.write(letter)
        sys.stdout.flush()
        time.sleep(a)

# This function removes numbers from an array
def remove_nums(arr): 
    # Declare a regular expression
    pattern = '[0-9]'  
    # Remove the pattern, which is a number
    arr = [re.sub(pattern, '', i) for i in arr]    
    # Return the array with numbers removed
    return arr

# This function cleans the passed in paragraph and parses it
def get_words(para):   
    # Split it into lower case    
    lower = para.lower().split()
    # Remove punctuation
    no_punctuation = (nopunc.translate(str.maketrans('', '', string.punctuation)) for nopunc in lower)
    # Remove integers
    no_integers = remove_nums(no_punctuation)
    # Remove stop words
    dirty_tokens = (data for data in no_integers if data not in stop_words)
    # Ensure it is not empty
    tokens = (data for data in dirty_tokens if data.strip())
    # Ensure there is more than 1 character to make up the word
    tokens = (data for data in tokens if len(data) > 1)
    
    # Return the tokens
    return tokens 

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

t0 = time.time()
# Sequence of prints for visual effects
print_slow(0.10, "Collecting Twitter Data for: tonykelly52...\n")
print_slow(0.10, "###########...\n")
print("DONE")
print()

print_slow(0.10, "Manipulating and TFxIDF vectorizing the data...\n")
print_slow(0.10, "#################################.............\n")
# Load in data
csv_table = pd.read_csv("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Project\\Code\\TonyTweets.csv")

# Create lists of columns we will use later
usernames = csv_table['username']
followings = csv_table['following']
followers = csv_table['followers']
is_bots = csv_table['is_bot']

# Make a reference to the csv_table
use_df = csv_table.copy()

# Create the overall corpus
s = pd.Series(use_df['tweet'])
corpus = s.apply(lambda s: ' '.join(get_words(s)))

# Load the vectorizer
vectorizer = load('D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Project\\Code\\vectorizer.joblib')
# Compute tfidf values
# This also updates the vectorizer
test = vectorizer.transform(corpus)

# Create a dataframe from the vectorization procedure
df = pd.DataFrame(data=test.todense(), columns=vectorizer.get_feature_names())

# Merge results into final dataframe
use_df.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)
result = pd.concat([use_df, df], axis=1, sort=False)
labels = result['is_bot']

# Time to load data
t1 = time.time()

# Delete unnecessary components from memory
del csv_table, use_df, df
gc.collect()

# Update
print_slow(0.05, "Loaded Data")
print()

# Change the dataset
result = result.drop('is_bot', axis=1)
result = result.drop('keep', axis=1)
result = result.drop('username', axis=1)

# https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas
#X_train, X_test, y_train, y_test = train_test_split(result[result.columns.difference(['tweet'])], labels, test_size = 0.2)


# Update
print_slow(0.05, "Loading Tweet Classifier (KNN) Model to predict tweets \n")
print_slow(0.05, "#################################.............\n")

# Create the models
knn = load("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Project\\Code\\compressed.pkl")

# Predict the models
knn_preds = knn.predict(result)

print_slow(0.05, "Predicted w/ KNN. Now summarizing account data...\n")

# Create a new data frame
knn_res = labels.to_frame('is_bot')
knn_res.insert(0, 'Prediction', knn_preds)
knn_res = usernames.to_frame().join(knn_res, how='inner')

# Count the number of tweet classifications per user
df_out = (knn_res.groupby(['username', 'Prediction']).is_bot.count().unstack(fill_value=0).
             rename({0: 'count_human', 1: 'count_bot'}, axis= 1))

# Determine the ratio of bot or not tweets
df_out['%_bot_tweets'] = df_out['count_bot'] / (df_out['count_bot'] + df_out['count_human'])

# Edit the dataframe
knn_res = followers.to_frame().join(knn_res, how='inner')
knn_res = followings.to_frame().join(knn_res, how='inner')

# Edit the dataframe
test_df = df_out.copy()
test_df = test_df.join(knn_res.set_index('username')[['following', 'followers']])

# Select the maximum followers and following
the_df = test_df.groupby(['username', 'count_bot', 'count_human', '%_bot_tweets'])['following', 'followers'].max()
the_df.to_csv("C:\\Users\\Kelly\\Desktop\\finaldf.csv")
a = pd.read_csv("C:\\Users\\Kelly\\Desktop\\finaldf.csv")
username = a['username']
un = username[0]
a = a.drop('username', axis=1)
print_slow(0.10, "Done. Now Loading Account Classifier (RF) Model to predict account \n")
rf = pickle.load(open("C:\\Users\\Kelly\\Desktop\\rf.pkl", 'rb'))

print("Here is the account summary: ")
print(a)
print()

rf_preds = rf.predict(a)
probability = np.max(rf.predict_proba(a))
the_prediction = rf_preds[0]
print_slow(0.15, "Predicting\n")
print_slow(0.15, "########################################..................\n")
print("DONE")
print()

if (the_prediction == 1):
    print("I classified: {} as a HUMAN with {} confidence".format(un, probability*100))
else:
    print("I classified: {} as a HUMAN with {} confidence".format(un, probability*100))

t1 = time.time()

# print(abs(t0-t1))

time.sleep(5)

import pyfiglet
ascii_text = pyfiglet.figlet_format("Tony Kelly's AI - Twitter Bot Prediction")
print(ascii_text)