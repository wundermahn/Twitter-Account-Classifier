import tweepy, json, pandas as pd, time
from pandas.io.json import json_normalize
from datetime import datetime

# https://stackoverflow.com/questions/58666135/how-to-extract-data-from-a-tweepy-object-into-a-pandas-dataframe

# Create TweetMiner class
class TweetMiner(object):

    # Set member data
    result_limit    =   20    
    data            =   []
    api             =   False

    # This would be specific for every user
    consumer_key =
    consumer_secret=
    access_token=
    access_token_secret=

    # Create dict of keys
    twitter_keys = {
        'consumer_key':        consumer_key,
        'consumer_secret':     consumer_secret,
        'access_token_key':    access_token,
        'access_token_secret': access_token_secret
    }


    # Constructor
    def __init__(self, keys_dict=twitter_keys, api=api, result_limit = 20):

        self.twitter_keys = keys_dict

        auth = tweepy.OAuthHandler(keys_dict['consumer_key'], keys_dict['consumer_secret'])
        auth.set_access_token(keys_dict['access_token_key'], keys_dict['access_token_secret'])

        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify = True)
        self.twitter_keys = keys_dict

        self.result_limit = result_limit

    # Function to mine user tweets
    def mine_user_tweets(self, user="dril", #BECAUSE WHO ELSE!
                         mine_rewteets=False,
                         max_pages=5):

        data           =  []
        last_tweet_id  =  False
        page           =  1

        # How many pages of tweets
        while page <= max_pages:
            # If its the last tweet
            if last_tweet_id:
                # Collect information
                statuses   =   self.api.user_timeline(screen_name = user,
                                                     count = self.result_limit,
                                                     max_id = last_tweet_id - 1,
                                                     tweet_mode = 'extended',
                                                     include_retweets=True
                                                    )        
            else:
                # Otherwise, continue to collect information
                statuses   =   self.api.user_timeline(screen_name = user,
                                                        count = self.result_limit,
                                                        tweet_mode = 'extended',
                                                        include_retweets = True)

            # Loop through the tweepy status objects
            for item in statuses:

                # Extract needed fields
                mined = {
                    'screen_name':     item.user.screen_name,
                    'text':            item.full_text,
                    'followers':       item.user.followers_count,
                    'following':       item.user.friends_count,
                    'source_device':   item.source,
                    'created_at':      item.created_at
                }

                # If someting was retweeted, include that here
                try:
                    mined['retweet_text'] = item.retweeted_status.full_text
                except:
                    mined['retweet_text'] = 'None'

                # Set the last tweet id to theone we just mined
                last_tweet_id = item.id
                data.append(mined)

            # Increase the page count
            page += 1

        # Return the data
        return data

# function to convert _json to JSON
def jsonify_tweepy(tweepy_object):
    # Turn it into a json
    json_str = json.dumps(tweepy_object._json)
    # jsonify it
    return json.loads(json_str)

#insert your Twitter keys here
consumer_key =''
consumer_secret=''
access_token=''
access_token_secret=''

# Authenticate yourself
auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
# Set access
auth.set_access_token(access_token, access_token_secret)
# Connet to the Twitter Dev API
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify = True)

# Get a list of users
users = []

# Try logging in
if(api.verify_credentials):
	print("Logged In Successfully")
else:
	print("Error -- Could not log in with your credentials")

# Find all of the followers for the account
followers = list(tweepy.Cursor(api.followers).items())

# Call the function and unload each _json into follower_list
followers_list = [jsonify_tweepy(follower) for follower in followers]

# Convert followers_list to a pandas dataframe
df = json_normalize(followers_list)

#df.to_csv("C:\\Users\\Kelly\\Desktop\\dataframe.csv")

# Collect 200 tweets from each user
miner = TweetMiner(result_limit = 200)
# Create a dict for the mined tweets
mined_tweets_dict = dict()

# Loop through each screen name
for name in df['screen_name'].unique():
    # Try and mine their tweets
    try:
        print("Trying: ", name)
        mined_tweets = miner.mine_user_tweets(user=name, max_pages=10)
        mined_tweets_dict[name] = pd.DataFrame(mined_tweets)
    except:
        print(name, " has protected tweets or private mode on ")

# Now open a file to write the data
with open('C:\\Users\\Kelly\\desktop\\final_tweets.csv', mode='a', encoding='utf-8', newline='') as f:
    # Write the mined tweets dict
    for i, df in enumerate(mined_tweets_dict.values()):
        if i == 0:
            df.to_csv(f, header=True, index=False)
        else:
            df.to_csv(f, header=False, index=False)