from datetime import datetime as dt
from datetime import timedelta
import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

document_order_df = pd.read_csv('document_order.csv')


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
        sent[n]

def main():
    return __name__ == "__main__"

def without_history():
    sent = sentiment_ndarray()
    tfidf = tfidf_matrix().todense()
    X = np.concatenate((sent,np.asarray(tfidf)),axis=1)
    return X

def with_history(days):
    sent = sentiment_ndarray()
    for row in sent:
        print(row)
    return None

def reproduction_rate_graph():
    covid = world_covid_df()
    dates = []
    rs = []
    for d,r in zip(covid['date'], covid['reproduction_rate']):
        dates.append(d)#.timestamp())
        rs.append(r)
    fig = plt.figure(figsize=(7,7))
    ax  = fig.add_subplot(111)
    ax.plot(dates,rs)
    ax.set_title('covid global reproduction rate')
    ax.set_xlabel('date (YYYY-MM)')
    ax.set_ylabel('reproduction rate (r)')
    plt.savefig('figure/reproduction_rate.png',dpi=300)

# return np.array with x[:,x.shape[1]-1] being the target value y
def label_data(dataset,n):
    zeros = np.zeros((6,dataset.shape[1]), dtype=int)
    dataset_zeros = np.append(zeros,dataset,axis=0)
    dataset_zeros = np.append(dataset_zeros,np.zeros((dataset_zeros.shape[0],8*6+1), dtype=int),axis=1)
    index = 6
    while index < dataset_zeros.shape[0]:
        ixgrid = np.ix_([i for i in range(index-6,index)], [j for j in range(8)])
        history = dataset_zeros[ixgrid]
        history = history.flatten()
        history_index= 0
        for value in history:
            dataset_zeros[index,dataset.shape[1]+history_index] = value
            history_index += 1
        if n+index < dataset_zeros.shape[0]:
            dataset_zeros[index,dataset.shape[1]+history_index] = dataset_zeros[n+index,0]
        else:
            dataset_zeros[index,dataset.shape[1]+history_index] = 0
        index += 1
    return dataset_zeros[6:]

if main():
    #reproduction_rate_graph()
    X = without_history()
    print(X[0])
    x_label = label_data(X,14)
    print(x_label[0])
