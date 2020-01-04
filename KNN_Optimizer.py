import pandas as pd, numpy as np, re, string, time, gc, pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

# Stop words from NLTK
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

# Collect n random tweets from each user
def collect_tweets(df, n):
    s = df.groupby('username').username.transform('count')
    new_df = df[s>=n].groupby('username').head(n).reset_index(drop=True)

    return new_df

t0 = time.time()

# Read in the two data files
not_bot = pd.read_csv("NotBot.csv", skiprows=1)
bot = pd.read_csv("Bot.csv", skiprows=1)

# Stack them
csv_table = pd.DataFrame(np.vstack((not_bot.values, bot.values)))

# Set columns
csv_table.columns = ['username', 'tweet', 'following', 'followers', 'is_retweet', 'is_bot']

# Create new dataframe selecting only 25 random results from each user
new_df = collect_tweets(csv_table, 25)

# Set list of names to exclude to build the test set
bot_names = ['BELOZEROVNIKIT', 	'ALTMANBELINDA', 	'666STEVEROGERS', 	'ALVA_MC_GHEE', 	'CALIFRONIAREP', 	'BECCYWILL', 	'BOGDANOVAO2', 	'ADELE_BROCK', 	'ANN1EMCCONNELL', 	'ARONHOLDEN8', 	'BISHOLORINE', 	'BLACKTIVISTSUS', 	'ANGELITHSS', 	'ANWARJAMIL22', 	'BREMENBOTE', 	'BEN_SAR_GENT', 	'ASSUNCAOWALLAS', 	'AHMADRADJAB', 	'AN_N_GASTON', 	'BLACK_ELEVATION', 	'BERT_HENLEY', 	'BLACKERTHEBERR5', 	'ARTHCLAUDIA', 	'ALBERTA_HAYNESS', 	'ADRIANAMFTTT']
not_bot_names = ['tankthe_hank', 	'cbars68', 	'megliebsch', 	'ZacharyFlair', 	'XavierRivera_', 	'NextLevel_Mel', 	'ChuckSpeaks_', 	'B_stever96', 	'Cassidygirly', 	'Sir_Fried_Alott', 	'msimps_15', 	'lasallephilo', 	'lovely_cunt_', 	'MisMonWEXP', 	'DurkinSays', 	'kdougherty178', 	'brentvarney44', 	'C_dos_94', 	'LSU_studyabroad', 	'Cyabooty', 	'PeterDuca', 	'chloeschultz11', 	'okweightlossdna', 	'hoang_le_96', 	'ShellMarcel']
bad_names = bot_names + not_bot_names

# Create a new dataframe of the good accounts
new_df['keep'] = new_df['username'].apply(lambda x: False if x in bad_names else True)
use_df = new_df[new_df['keep'] == True]
test_df = new_df[new_df['keep'] == False]

# Remove stuff we no longer need from memory
del new_df, csv_table, bad_names
gc.collect()

# Write the test data to file to use
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

print("Loaded data")

# https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas
#X_train, X_test, y_train, y_test = train_test_split(result[result.columns.difference(['is_bot', 'username', 'tweet', 'keep'])], labels, test_size = 0.2)

# Create training data
training_data = result[result.columns.difference(['is_bot', 'username', 'tweet', 'keep'])]

# Remove stuff from memory
del result
gc.collect()

# Update
print("Split data")

# Create the models
knn = KNeighborsClassifier(n_neighbors = 7, weights='uniform', metric='euclidean', algorithm='auto')

# Train the models
knn.fit(training_data, labels)

# Update
print("Trained KNN")

# Serialize, compress, and dump the vectorizer and the model
dump(vectorizer, "C:\\Users\\J39304\\Desktop\\Initial Classifier Test\\vectorizer.joblib")
dump(knn, 'C:\\Users\\J39304\\Desktop\\Initial Classifier Test\\compressed.pkl.z', compress=7)

# Update
print("Script Finished")