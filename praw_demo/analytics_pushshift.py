import pandas as pd
import datetime
from datetime import datetime as dt
from pprint import pprint as pp

today = dt.today()

first_day = dt(year=2020,month=1,day=1)

def days_ago(n):
    return int((today - datetime.timedelta(days=n)).timestamp())

def before(n):
    return days_ago(n-1)

def after(n):
    return days_ago(n)

def date(unix_time):
    return dt.utcfromtimestamp(unix_time).date()

def working_datetime(d):
    return dt(year=d.year,month=d.month,day=d.day)

def to_dict(d,comments_df):
    return {'number_of_comments': len(comments_df.index), 'date': d}
df = pd.DataFrame()
total_number_of_comments = 0

n = 0
d = date(days_ago(n))
number_of_valid_dates = 0
while working_datetime(d) > first_day:
    d = date(days_ago(n))
    n += 1
    print(d)
    try:
        comments_df = pd.read_csv(f'pushshift/{d}.csv')
    except FileNotFoundError as e:
        print("FILE_DOES_NOT_EXIST::SKIPPING::", d)
        print(e)
        continue
    except pd.errors.EmptyDataError as e:
        print("NO_DATA::SKIPPING::", d)
        print(e)
        continue
    print('WORKING...')
    df = df.append(to_dict(d,comments_df),ignore_index=True)
    number_of_valid_dates += 1
    total_number_of_comments += len(comments_df.index)

df.head(20)
df.to_csv('r_coronavirus_analytics/number_of_comments.csv')
print('TOTAL NUMBER OF COMMENTS:\t', total_number_of_comments)
print('AVERAGE NUMBER OF COMMENTS:\t', total_number_of_comments/number_of_valid_dates)
print('FINISHED')
