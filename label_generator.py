from datetime import datetime as dt
from datetime import timedelta
import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt



def load_features():
    return None

def name(datetime):
    return f'r/cleaned/text_data_{datetime.date()}.csv'

first_day = dt(year=2020, month=6, day=15)

def filenames_earliest_to_latest():
    d = first_day 
    end_date = dt(year=2021, month=11, day=30)
    while d < end_date:
        n = name(d)
        if os.path.exists(n):
            yield d
        else:
            print("ERROR: FILE MISSING: ", d)
            exit()
        d += timedelta(days=1)

def document_order():
    return pd.read_csv('document_order.csv')

document_order_df = document_order()

def tfidf_matrix():
    return pickle.load(open('tf_idf_weighted_document_term_matrix.pickle','rb'))

def tfidf_df():
    matrix = tfidf_matrix()
    print(type(matrix))

sentiment_columns = ["new_cases","d_s_pos","d_s_neg","total","por_pos","por_neg","pos_strength","neg_strength"]
def sentiment_df():
    return pd.read_csv('praw_demo/collated_covid_sentiment.csv',index_col=False,usecols=sentiment_columns)

def sentiment_ndarray():
    return sentiment_df().to_numpy()

def world_covid_df():
    df = pd.read_csv('praw_demo/world_covid_data.csv')
    df['date'] = pd.to_datetime(df['date'], dayfirst=True,errors='raise')
    return df.sort_values(by='date')

def generate_labels():
    yield None

def sentiment(n,sent):
    if (n < 0):
        return [0,0,0,0, 0,0,0,0]
    else:
        return sent[n]

def with_history(days_back):
    sent = sentiment_ndarray()
    tfidf = tfidf_matrix()
    with_history = np.zeros(shape=(len(document_order_df['0']), days_back*sent.shape[1] + tfidf.shape[1]), dtype=np.float64)
    for i in range(with_history.shape[0]):
        stride = sent.shape[1]
        for n in range(days_back):
            if i-n >= 0:
                with_history[i,stride*n:stride*(n+1)] = sentiment(i-n,sent)
        with_history[i,stride*(days_back):with_history.shape[1]] = tfidf[i].todense()
    return with_history
    ## MANUAL VALIDATION
    ## for i in range(days_back):
    ##     print(i,"\t",with_history[0, i*sent.shape[1]:(i+1)*sent.shape[1]])
    ## print(sent[0])

    ## print(with_history[0, 0:(days_back+1)*sent.shape[1]])
    ## for i in range(sent.shape[1]):
    ##     print(tfidf[0,i])


def without_history():
    sent = sentiment_ndarray()
    tfidf = tfidf_matrix()
    X = np.append(sent,tfidf.todense())
    X = np.reshape(X, (tfidf.shape[0], tfidf.shape[1]+sent.shape[1]))
    return X

def reproduction_rate_graph():
    covid = world_covid_df()
    dates = []
    rs = []
    for d,r in zip(covid['date'], covid['reproduction_rate']):
        dates.append(d)
        rs.append(r)
    fig = plt.figure(figsize=(7,7))
    ax  = fig.add_subplot(111)
    ax.plot(dates,rs)
    ax.set_title('covid global reproduction rate')
    ax.set_xlabel('date (YYYY-MM)')
    ax.set_ylabel('reproduction rate (r)')
    plt.savefig('figure/reproduction_rate.png',dpi=300)

def main():
    return (__name__ == "__main__")

if main():
    X = with_history(7)
    print(X.shape)
