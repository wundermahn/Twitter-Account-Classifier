import pandas as pd, numpy as np, re, string, time, gc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from joblib import dump

#NLTK Stop Words
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

# Collect the top 25 tweets from each user
def collect_tweets(df, n):
    # Create a new series that counts the number of tweets per username
    s = df.groupby('username').username.transform('count')
    # Create a new df with only accounts that have more than 25, and only the first 25
    new_df = df[s>=n].groupby('username').head(n).reset_index(drop=True)

    # Return it
    return new_df

t0 = time.time()

# Read in the datasets and perform stacking operations to create thed ataset
not_bot = pd.read_csv("NotBot.csv", skiprows=1)
bot = pd.read_csv("Bot.csv", skiprows=1)
csv_table = pd.DataFrame(np.vstack((not_bot.values, bot.values)))
csv_table.columns = ['username', 'tweet', 'following', 'followers', 'is_retweet', 'is_bot']

# Create the smaller dataset with only the top 25 tweets
new_df = collect_tweets(csv_table, 25)

# Create a list of test accounts to remove
bot_names = ['BELOZEROVNIKIT', 	'ALTMANBELINDA', 	'666STEVEROGERS', 	'ALVA_MC_GHEE', 	'CALIFRONIAREP', 	'BECCYWILL', 	'BOGDANOVAO2', 	'ADELE_BROCK', 	'ANN1EMCCONNELL', 	'ARONHOLDEN8', 	'BISHOLORINE', 	'BLACKTIVISTSUS', 	'ANGELITHSS', 	'ANWARJAMIL22', 	'BREMENBOTE', 	'BEN_SAR_GENT', 	'ASSUNCAOWALLAS', 	'AHMADRADJAB', 	'AN_N_GASTON', 	'BLACK_ELEVATION', 	'BERT_HENLEY', 	'BLACKERTHEBERR5', 	'ARTHCLAUDIA', 	'ALBERTA_HAYNESS', 	'ADRIANAMFTTT']
not_bot_names = ['tankthe_hank', 	'cbars68', 	'megliebsch', 	'ZacharyFlair', 	'XavierRivera_', 	'NextLevel_Mel', 	'ChuckSpeaks_', 	'B_stever96', 	'Cassidygirly', 	'Sir_Fried_Alott', 	'msimps_15', 	'lasallephilo', 	'lovely_cunt_', 	'MisMonWEXP', 	'DurkinSays', 	'kdougherty178', 	'brentvarney44', 	'C_dos_94', 	'LSU_studyabroad', 	'Cyabooty', 	'PeterDuca', 	'chloeschultz11', 	'okweightlossdna', 	'hoang_le_96', 	'ShellMarcel']
bad_names = bot_names + not_bot_names

# Now make a new dataframe removing the to-be test accounts
new_df['keep'] = new_df['username'].apply(lambda x: False if x in bad_names else True)
use_df = new_df[new_df['keep'] == True]
test_df = new_df[new_df['keep'] == False]

# Remove unnecessary garbage from memory
del new_df, csv_table, bad_names
gc.collect()

# Write this to file
test_df.to_csv("C:\\Users\\J39304\\Desktop\\Initial Classifier Test\\test_data_set.csv")

# Create the overall corpus
s = pd.Series(use_df['tweet'])
corpus = s.apply(lambda s: ' '.join(get_words(s)))

# Create a vectorizer
vectorizer = TfidfVectorizer()
# Compute tfidf values
# This also updates the vectorizer
test = vectorizer.fit_transform(corpus)

# Create a dataframe from the vectorization procedure
df = pd.DataFrame(data=test.todense(), columns=vectorizer.get_feature_names())

# Merge results into final dataframe
use_df.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)
result = pd.concat([use_df, df], axis=1, sort=False)
labels = list(result['is_bot'])

# Time to load data
t1 = time.time()

# Update
print("Loaded data")

# https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas
X_train, X_test, y_train, y_test = train_test_split(result[result.columns.difference(['is_bot', 'username', 'tweet', 'keep'])], labels, test_size = 0.2)

# Update
print("Split data")

# Time to split data
t2 = time.time()

# Create the models
knn = KNeighborsClassifier(n_neighbors = 7)
nbc = MultinomialNB()
rf = RandomForestClassifier()
lr = LogisticRegression()

# Train the models
knn.fit(X_train, y_train)
nbc.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Update
print("Trained Models")

# Time to train classifiers
t3 = time.time()

# Predict the models
knn_preds = knn.predict(X_test)
nbc_preds = nbc.predit(X_test)
rf_preds = rf.predict(X_test)
lr_preds = lr.predict(X_test)

# Update
print("Predicted Data")

# Time to test classifiers
t4 = time.time()

# Collect metrics
knn_precision, knn_recall, knn_f1 = collect_metrics(knn_preds, y_test)
nbc_precision, nbc_recall, nbc_f1 = collect_metrics(nbc_preds, y_test)
rf_precision, rf_recall, rf_f1 = collect_metrics(rf_preds, y_test)
lr_precision, lr_recall, lr_f1 = collect_metrics(lr_preds, y_test)

# Pretty print the results
print("KNN     | Recall: {} | Precision: {} | F1: {}".format(knn_recall, knn_precision, knn_f1))
print("NBC     | Recall: {} | Precision: {} | F1: {}".format(nbc_recall, nbc_precision, nbc_f1))
print("RF      | Recall: {} | Precision: {} | F1: {}".format(rf_recall, rf_precision, rf_f1))
print("LR      | Recall: {} | Precision: {} | F1: {}".format(lr_recall, lr_precision, lr_f1))

# Time to pretty print and collect results
t5 = time.time()

# Print times
print()
print("Time to load data: {}".format(abs(t0-t1)))
print("Time to split data: {}".format(abs(t1-t2)))
print("Time to train classifiers {}".format(abs(t2-t3)))
print("Time to test classifiers: {}".format(abs(t3-t4)))
print("Time to pretty print results: {}".format(abs(t5-t4)))

# dump(knn, "C:\\Users\\J39304\\Desktop\\Initial Classifier Test\\knn_model.joblib")
# dump(vectorizer, "C:\\Users\\J39304\\Desktop\\Initial Classifier Test\\vectorizer.joblib")