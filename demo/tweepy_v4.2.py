import tweepy
from datetime import datetime

toDate = datetime.fromisoformat('2020-01-01')

cork_coords = "point_radius:[51.88532757350978 -8.467862308148838 25mi]"
#(year=datetime.year(2020),month=datetime.month(1),day=datetime.day(1),hour=datetime.hour(23), minute=datetime.minute(59), second=datetime.second(59))
print(toDate.utcoffset())

print(f'tweepy version {tweepy.__version__}')

api_key = "VSIB59ufJlbnErA106Z9C24VK"
api_key_secret = "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX"
consumer_secret = "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6"
auth = tweepy.OAuthHandler("VSIB59ufJlbnErA106Z9C24VK", "OIGv5ohnTdnlM91pubPHjt52YMSvrobHMicBPtdkz8h83gT5hX")
auth.set_access_token("1051212950432833536-Z2KtGhRjhd7svXFQi8UyDZq0zKp0VE", "vGQee7KZ9peaSIxkpY4AQqYOj4gp3j6qEkf4a69blPRM6")

api = tweepy.API(auth)
client = tweepy.Client(auth,wait_on_rate_limit=True)


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

tweets = api.search_full_archive(label="oisinthomasmorrin",query=f"(covid OR corona OR virus OR party OR club OR gather) {cork_coords}")


for tweet in tweets:
    print(f"\n\n########################################## {tweet.author.name}")
    print(tweet.text)
    # for prop, val in vars(tweet).items():
    #     print(prop)

print(len(tweets))


#  ### API v2
#  for tweet in client.search_recent_tweets(query="covid", end_time="2021-01-01T23:59:59Z"):
#      print(f"\n\n########################################## {tweet.author}")
#      print(tweet.text)

