#!/home/neimhin/miniconda3/envs/pond/bin/python
import datetime
from datetime import datetime as dt
import praw
import pickle
import pprint
import sys
import pandas as pd
from pmaw import PushshiftAPI


api = PushshiftAPI()

today = dt(year=2020,month=7,day=9)

def days_ago(n):
    return int((today - datetime.timedelta(days=n)).timestamp())

def before(n):
    return days_ago(n-1)

def after(n):
    return days_ago(n)

def date(unix_time):
    return dt.utcfromtimestamp(unix_time).date()

subreddit = "coronavirus"
limit=100000

n = 0
while True:
    n += 1
    retry = True
    while retry:
        try:
            comments = api.search_comments(subreddit=subreddit, limit=limit, before=before(n), after=after(n))
            comments_df = pd.DataFrame(comments)
            comments_df.head(5)
            d = date(days_ago(n))
            comments_df.to_csv(f'pushshift/{d}.csv',header=True,index=False,columns=list(comments_df.axes[1]))
            retry = False
        except Exception as e:
            print(e)
