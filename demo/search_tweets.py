import tweepy
from datetime import datetime
from datetime import date as pdate
import json
import time
import pandas as pd
import pickle
import traceback

def raw_coordinates():
    coord_df = pd.read_csv('ie.csv')
    def coordinates_alt():
        return coord_df
    global raw_coordinates
    raw_coordinates = coordinates_alt
    return coord_df

def coords_str_api2(coords_tuple):
    # cork_coords = "point_radius:[51.88532757350978 -8.467862308148838 400km]"
    return f'point_radius:[{coords_tuple[0]} {coords_tuple[1]} 25mi]'

def coords_str_api1(coords_tuple):
    # cork_coords = "point_radius:[51.88532757350978 -8.467862308148838 400km]"
    return f'{coords_tuple[0]},{coords_tuple[1]},25mi'

def coordinates():
    return map(coords_str_api1, zip(raw_coordinates()['lat'],raw_coordinates()['lng']))


print(f'tweepy version {tweepy.__version__}')

tweepy.RateLimitError = tweepy.TooManyRequests

def load_api_object():
    # consumer_key = "VSIB59ufJlbnErA106Z9C24VK"
    # consumer_secret = "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX"
    # access_token = "1051212950432833536-Z2KtGhRjhd7svXFQi8UyDZq0zKp0VE"
    # access_token_secret = "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6"
    # 
    # auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # auth.set_access_token(access_token, access_token_secret)
    # 
    # api = tweepy.API(auth, wait_on_rate_limit=True)
    # return api
    return pickle.load(open('neimhin_api_object.pickle','rb'))

api = load_api_object() 

def limit_handled(cursor):
    while True:
        try:
            yield next(cursor)
        except tweepy.TooManyRequests:
            print("waiting 15*60 seconds ...")
            time.sleep(15*60)

def filename(city):
    return f'tweets_by_city/{city}.pickle'

def db(city):
    import os
    if not os.path.exists(filename(city)):
       return dict() 
    return pickle.load(open(filename(city), 'rb'))

def dump(data, city):
    pickle.dump(data, open(filename(city), 'wb'))

def insert(city,date,tweets):
    import os
    database = dict()
    if os.path.exists(filename(city)):
        database = db(city) 
    if date in database:
        database[date] += tweets 
    else:
        database[date] = tweets
    dump(database, city)

def show_city(city):
    db = dict(pickle.load(open(filename(city),'rb')))
    for d in db:
        print(f'{d}\t{db[d]}')

# insert('neverland', str(date.today()), [1,2,3,4])
# show_city('neverland')

def extract_date(r):
    date = r['date'].split('-')
    date = date[2] + '/' + date[1] + '/' + date[0]
    toDate = r["date"] # .replace("-","") # + "0000" # not how it should be written
    return (date,toDate)

def cities():
    return raw_coordinates()['city']

def dates():
    covid_df = pd.read_csv("ireland_covid_data.csv")
    for _,r in covid_df.iterrows():
        yield(extract_date(r))

if __name__ == "__main__":
    fromDate = "202002280000"
    for date, toDate in dates(): # iterate over dates
        date = str(pdate.today())
        print(date)
        retry=True
        while retry:
            try:
                for city, coords in zip(cities(), coordinates()): #iterate over cities
                    print(f"####### city: {city}\tdate: {date}")
                    if date in db(city):
                        retry = False
                        break
                    tweets = api.search_tweets(
                        # label="oisinthomasmorrin",
                        q=f"covid",
                        geocode=coords,
                        until=date)

                        #    ,
                        #fromDate=fromDate,
                        #toDate=toDate)

                    for tweet in tweets:
                        print(tweet._json)

                    insert(city, date, tweets)
                retry = False

            # In case of exception we retry
            except tweepy.errors.TooManyRequests:
                traceback.print_exc()
                print("waiting 15*60 seconds ...")
                time.sleep(15*60)
                retry = True
                print("retrying")
                continue
