import tweepy
from datetime import datetime
import json
import time
import pandas as pd


coordinates_df = pd.read_csv("ie.csv")
print(coordinates_df.head())
coordinates_df["coords"] = coordinates_df["lat"].astype(str) + "," + coordinates_df["lng"].astype(str)+",25mi"
coordinates_df = pd.read_csv("ireland_covid_data.csv")
print(coordinates_df.head())

covid_df = pd.read_csv("ireland_covid_data.csv")
print(covid_df.head())

toDate = datetime.fromisoformat('2020-01-01')

cork_coords = "point_radius:[51.88532757350978 -8.467862308148838 400km]"
cork_coords_api1 = "51.88532757350978,-8.467862308148838,25mi"
# (year=datetime.year(2020),month=datetime.month(1),day=datetime.day(1),hour=datetime.hour(23), minute=datetime.minute(59), second=datetime.second(59))
print(toDate.utcoffset())

print(f'tweepy version {tweepy.__version__}')

consumer_key = "VSIB59ufJlbnErA106Z9C24VK"
consumer_secret = "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX"
access_token = "1051212950432833536-Z2KtGhRjhd7svXFQi8UyDZq0zKp0VE"
access_token_secret = "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweepy.RateLimitError = tweepy.TooManyRequests


def limit_handled(cursor):
    while True:
        try:
            yield next(cursor)
        except tweepy.TooManyRequests:
            print("waiting 15*60 seconds ...")
            time.sleep(15*60)


if __name__ == "__main__":
    #for tweet in limit_handled(tweepy.Cursor(
    #        api.search_tweets,
    #        geocode=rowcork_coords,
    #        q='covid').items()):
    #    with open("demo_twitter_data/" + str(tweet.created_at.date()), 'a') as f:
    #        f.write(json.dumps(tweet._json, indent=4) + ',')
    big_dataset = {}
    lastDate= "202002280000"
    for i,r in covid_df.iterrows(): # iterate over dates
        tweets_data = []
        covid_tweets = 0
        for index, row in coordinates_df.iterrows(): #iterate over cities
            coordinates = row["coords"]
            today = r["date"].replace("-","") + "0000" # not how it should be written
            date = r["date"].replace("-","/")[::-1] #use DD/MM/YYYY for dataset
            tweets = api.search_full_archive(
                label="oisinthomasmorrin",
                q=f"covid OR corona OR virus {coordinates}", fromDate=lastDate,toDate=today)
            tweets_data += tweets # not how it should be written
            covid_tweets = len(tweets_data)
        if len(tweets_data) < 1000: #add extra tweets from general pool ono that day
            for index, row in coordinates_df.iterrows(): # iterate over cities
                coordinates = row["coords"]
                tweets = api.search_full_archive(
                    label="oisinthomasmorrin",
                    q=f"{coordinates}", fromDate=lastDate,toDate=today)
                tweets_data += tweets
                if len(tweets_data) >= 1000:
                    break
        lastDate = today
        big_dataset[date] = {"tweets":" ".join(tweets_data),"covid_tweets":covid_tweets}

    df = pd.DataFrame(big_dataset,columns=["tweets","covid_tweets"])
    df.to_csv("tweets_for_dates_ie.csv")
    # # Write property names of tweets to a file
    # for tweet in limit_handled(tweepy.Cursor(api.search_tweets, q='covid').items()):
    #     for name in vars(tweet):
    #         with open("tweet_property_names.md","a") as f:
    #             f.write(name + '\n')
    #     break

# for tweet in tweepy.Cursor(api2.search_tweets, q='covid').items():
#     print(tweet.text)


# API v1.1

# # HOME TIMELINE
# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print("\n\n##########################################")
#     print(tweet.text)


# # USER DETAILS
# neimhin = api.get_user(screen_name='neimhin')
#
# print(neimhin.screen_name)
# print(neimhin.followers_count)
# for friend in neimhin.friends():
#     print(friend.screen_name)


# tweets = api.search_tweets(q="covid")
#
# for tweet in tweets:
#     print(f"\n\n########################################## {tweet.author.name}")
#     print(tweet.text)
#     for prop, val in vars(tweet).items():
#         print(prop)
#
# print(len(tweets))

#
#
#
# for tweet in tweets:
#     print(f"\n\n########################################## {tweet.author.name}")
#     print(tweet.text)
#     # for prop, val in vars(tweet).items():
#     #     print(prop)


#  ### API v2
#  for tweet in client.search_recent_tweets(query="covid", end_time="2021-01-01T23:59:59Z"):
#      print(f"\n\n########################################## {tweet.author}")
#      print(tweet.text)
