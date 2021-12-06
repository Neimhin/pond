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

def word_count():
    return pd.read_csv('cleaned_text_word_counts.csv')

number_dates_with_data = 0
total_char_count = 0

def to_dict(d,comments_df):
    l = comments_df['text']
    l = len(str(l))
    print(l)
    if l > 0:
        global number_dates_with_data
        number_dates_with_data += 1
    global total_char_count
    total_char_count += l
    return {'char_count': l, 'date': d}

def old():
    df = pd.DataFrame()
    total_number_of_comments = 0
    
    n = 0
    d = date(days_ago(n))
    while working_datetime(d) > first_day:
        d = date(days_ago(n))
        n += 1
        print(d)
        try:
            comments_df = pd.read_csv(f'r/cleaned/text_data_{d}.csv').astype('string')
            print(comments_df)
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
        total_number_of_comments += len(comments_df.index)
    
    print('TOTAL_CHAR_COUNT:\t', total_char_count)
    print('AVERAGE_CHAR_COUNT:\t', float(total_char_count)/number_dates_with_data)
    print('NUMBER OF DATES WITH DATA:\t', number_dates_with_data)
    df.head(20)
    df.to_csv('cleaned_analytics.csv')
    print('FINISHED')

if __name__ == '__main__':
    wc = word_count()
    print(sum(wc['word']))
    print(len(wc['word']))
    print(float(sum(wc['word']))/len(wc['word']))
