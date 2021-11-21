import tweepy
import pickle

consumer_key = "VSIB59ufJlbnErA106Z9C24VK"
consumer_secret = "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX"
access_token = "1051212950432833536-Z2KtGhRjhd7svXFQi8UyDZq0zKp0VE"
access_token_secret = "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

pickle.dump(api, open('api_object.pickle','wb'))
