# -*- coding: utf-8 -*-
import tweepy
print(tweepy.__version__)
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
import pandas as pd
import re
import time
import numpy as np

#if you are going to run this, please change the consumer key and token, oath token and token secret and directory for the csv files. You may also change the max files to be made.
if __name__ == "__main__":
    count = 0
    # Uh oh, don't overwrite global functions!!!
    # max = 1
    ## This fails if you set max = 1
    ## print(max(0, 1))
    max_calls = 1
    while True:
        number = count
        #set authorisation based on token and key
        auth = OAuthHandler("VSIB59ufJlbnErA106Z9C24VK", "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX")
        auth.set_access_token("1051212950432833536-Z2KtGhRjhd7svXFQi8UyDZq0zKp0VE", "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6")

        api = tweepy.API(auth)
        
        #max number of calls
        if number == max_calls:
            break
    
        count += 1
        try:
            #call to API for tweets within the last week that were in English
            # tweets = tweepy.Cursor(api.search_tweets, q="bitcoin", fromDate="202101010000",toDate="202101020000",lang = 'en')
            tweets = api.search_geo(query='covid', lat=53.39845, long=-6.16178, granularity='country')
            for status in tweets.items():
                print(status._json["text"])

        except:
            continue
        # Looks like tweepy.error.RateLimitError isn't in version 4??
        #catches error when rate limit of tweets reached before continuing after cooldown period
        #except tweepy.errors.TweepyException:
        #    print("stopping for 15mins...")
        #    time.sleep(60 * 15)
        #    print("starting again after cool down.")
        #    continue

        # {"x":3000, ...... x5..., y=}
