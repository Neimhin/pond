# -*- coding: utf-8 -*-
from tweepy import OAuthHandler
from tweepy import API
from tweepy.streaming import StreamListener
from tweepy import Stream
from tweepy import error
import tweepy
import pandas as pd
import re
import time
import numpy as np

#if you are going to run this, please change the consumer key and token, oath token and token secret and directory for the csv files. You may also change the max files to be made.
if __name__ == "__main__":
    count = 0
    max = 1
    while True:
        number = count
        #set authorisation based on token and key
        auth = OAuthHandler("VSIB59ufJlbnErA106Z9C24VK", "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX")
        auth.set_access_token("1051212950432833536-Z2KtGhRjhd7svXFQi8UyDZq0zKp0VE", "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6")

        api = API(auth)
        
        #max number of calls
        if number == max:
            break
    
        count += 1
        try:
            #call to API for tweets within the last week that were in English
            tweets = tweepy.Cursor(api.search, q="bitcoin", fromDate="202101010000",toDate="202101020000",lang = 'en')
            for status in tweets.items():
                print(status._json["text"])
        #catches error when rate limit of tweets reached before continuing after cooldown period
        except error.RateLimitError:
            print("stopping for 15mins...")
            time.sleep(60 * 15)
            print("starting again after cool down.")
            continue

        # {"x":3000, ...... x5..., y=}
