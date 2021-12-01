import pandas as pd
import datetime
from datetime import datetime as dt

today = dt.today()

first_day  = dt(year=2020,month=11,day=20)

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

n = 0
d = date(days_ago(n))
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

print('FINISHED')
