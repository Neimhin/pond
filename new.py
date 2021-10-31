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
import sys

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


#if you are going to run this, please change the consumer key and token, oath token and token secret and directory for the csv files. You may also change the max files to be made.
if __name__ == "__main__":
        count = 1
        max = 30
        last_id = None
        print(tweepy.__version__)
        while True:
        #try:
                #number = count
                #set authorisation based on token and key
                auth = OAuthHandler("VSIB59ufJlbnErA106Z9C24VK", "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX")
                auth.set_access_token("1051212950432833536-Z2KtGhRjhd7svXFQi8UyDZq0zKp0VE", "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6")
                api = API(auth, wait_on_rate_limit=True)                #max number of calls
                if count == max:
                    break
                
                
                #try:
                    #call to API for tweets within the last week that were in English
                d1 = f"2021-03-0{count}:0000"
                count_2 = count+1
                d2 = f"2021-03-0{count+1}:0000"
                results = [status._json for status in tweepy.Cursor(api.search, q="covid",until=d2, lang = 'en',geocode="53.1424,7.6921,280km").items(50)]
    # Now yo    u can iterate over 'results' and store the complete message from each tweet.
                my_tweets = []

                for result in results:
                    try:
                        my_tweets.append(result["full_text"])
                    except:
                        my_tweets.append(result["retweeted_status"]["full_text"])

                count += 1
                print(f"{count} {len(my_tweets)}")
                print(my_tweets)
            #catches error when rate limit of tweets reached before continuing after cooldown period
            #except:
            #    print("stopping for 15mins...")
            #    i = 0
            #    while i < 60 * 15:
            #        progress(i, 900)
            #        time.sleep(1)
            #        i += 1
            #    print("starting again after cool down.")
            #    continue
            #    # {"x":3000, ...... x5..., y=}