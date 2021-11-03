import tweepy
from datetime import datetime
import json
import time

toDate = datetime.fromisoformat('2020-01-01')

cork_coords = "point_radius:[51.88532757350978 -8.467862308148838 400km]"
cork_coords_api1 = "51.88532757350978,-8.467862308148838,25mi"
#(year=datetime.year(2020),month=datetime.month(1),day=datetime.day(1),hour=datetime.hour(23), minute=datetime.minute(59), second=datetime.second(59))
print(toDate.utcoffset())

print(f'tweepy version {tweepy.__version__}')

consumer_key = "VSIB59ufJlbnErA106Z9C24VK"
consumer_secret = "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX"
access_token = "1051212950432833536-Z2KtGhRjhd7svXFQi8UyDZq0zKp0VE"
access_token_secret= "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6"

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
    for tweet in limit_handled(tweepy.Cursor(
        api.search_tweets,
        geocode=cork_coords_api1,
        q='covid').items()):
        with open("demo_twitter_data/" + str(tweet.created_at.date()), 'a') as f:
            f.write(json.dumps(tweet._json, indent=4) + ',')
    

    # # Write property names of tweets to a file
    # for tweet in limit_handled(tweepy.Cursor(api.search_tweets, q='covid').items()):
    #     for name in vars(tweet):
    #         with open("tweet_property_names.md","a") as f:
    #             f.write(name + '\n')
    #     break

# for tweet in tweepy.Cursor(api2.search_tweets, q='covid').items():
#     print(tweet.text)


### API v1.1

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

# tweets = api.search_tweets(
#         # label="oisinthomasmorrin",
#         q=f"covid corona virus" # "(covid OR corona OR virus OR party OR club OR gather) {cork_coords}")
#         )
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

