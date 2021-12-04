from datetime import datetime as dt
from datetime import timedelta
import os
import numpy as np
import pickle
import pandas as pd

document_order_df = pd.read_csv('document_order.csv')


def load_features():
    return None

def name(datetime):
    return f'r/cleaned/text_data_{datetime.date()}.csv'

def filenames_earliest_to_latest():
    d = dt(year=2020, month=6, day=15)
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

def generate_labels():
    yield None

def main():
    return __name__ == "__main__"

if main():
    sent = sentiment_ndarray()
    tfidf = tfidf_matrix()
    print(tfidf.shape)
    print(sent.shape)
    print(tfidf.ndim)
    print(sent.ndim)
    X = np.append(sent,tfidf.todense())
    X = np.reshape(X, (tfidf.shape[0], tfidf.shape[1]+sent.shape[1]))
    print(X.shape)
