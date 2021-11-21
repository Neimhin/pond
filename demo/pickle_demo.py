import tweepy
from datetime import datetime
from datetime import date
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
    return pickle.load(open('oisin_api_object.pickle','rb'))

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
    toDate = r["date"].replace("-","") + "0000" # not how it should be written
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
        retry=True
        print(f"####### {date}")
        while retry:
            try:
                for city, coords in zip(cities(), coordinates()): #iterate over cities
                    if date in db(city):
                        retry = False
                        break
                    tweets = api.search_full_archive(
                        label="oisinthomasmorrin",
                        query=f"covid OR corona OR virus {coords}",
                        fromDate=fromDate,
                        toDate=toDate)

                    insert(city, date, tweets)
                retry = False

            # In case of exception we retry
            except tweepy.errors.TooManyRequests:
                traceback.print_exc()
                print("waiting 15*60 seconds ...")
                time.sleep(15*60)
                # retry = True
                print("retrying")
                continue
