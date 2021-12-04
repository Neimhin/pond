import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime as dt
from datetime import timedelta

def name(datetime):
    return f'r/cleaned/text_data_{datetime.date()}.csv'

def filenames_earliest_to_latest():
    d = dt(year=2020, month=1, day=1)
    end_date = dt(year=2021, month=12, day=31)
    while d < end_date:
        n = name(d)
        if os.path.exists(n):
            yield n
        d += timedelta(days=1)

def documents_earliest_to_latest():
    document_order = []
    for filename in filenames_earliest_to_latest():
        document_order.append(filename)
        df = pd.read_csv(filename)
        yield df['text'][0]
    pd.DataFrame(document_order).to_csv('document_order.csv')

number_of_documents = len(list(os.listdir('r/cleaned/')))

def build():
    vectorizer = TfidfVectorizer(
            input='filename',
            ngram_range=(1,1),
            max_df = 0.75,
            min_df=int(0.1*number_of_documents),
            stop_words='english')
    X = vectorizer.fit_transform(filenames_earliest_to_latest())
    pickle.dump(vectorizer, open('tfidf_all.pickle','wb'))
    pickle.dump(X, open('tf_idf_weighted_document_term_matrix.pickle','wb'))
    return (X, vectorizer)

def read():
    return (pickle.load(open('tf_idf_weighted_document_term_matrix.pickle','rb')), pickle.load(open('tfidf_all.pickle','rb')))

if __name__ == '__main__':
    (X,vectorizer) = build()
    print(X.shape)
