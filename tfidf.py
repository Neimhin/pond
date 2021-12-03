import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime as dt
from datetime import timedelta

# filenames = list(map(lambda x: 'r/coronavirus/' + x, os.listdir('r/coronavirus/')))
def name(datetime):
    return f'r/coronavirus_body/{datetime.date()}.csv'

def filenames_earliest_to_latest():
    d = dt(year=2020, month=6, day=14)
    end_date = dt(year=2021, month=12, day=31)
    while d < end_date:
        n = name(d)
        if os.path.exists(n):
            yield n
        d += timedelta(days=1)

def build():
    vectorizer = TfidfVectorizer(input='filename', ngram_range=(1,2), max_df = 0.8, min_df=5,stop_words='english')
    X = vectorizer.fit_transform(filenames_earliest_to_latest())
    pickle.dump(vectorizer, open('tfidf_all.pickle','wb'))
    pickle.dump(X, open('tf_idf_weighted_document_term_matrix.pickle','wb'))
    return (X, vectorizer)

def read():
    return (pickle.load(open('tf_idf_weighted_document_term_matrix.pickle','rb')), pickle.load(open('tfidf_all.pickle','rb')))

if __name__ == '__main__':
    (X,vectorizer) = read()
    print(X.shape)
