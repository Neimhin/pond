from datetime import datetime as dt
from datetime import timedelta
import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import math

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

def concatenate_data():
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
def make_n_day_prediction_dataset(dataset,n):
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
            dataset_zeros[index,dataset.shape[1]+history_index] = dataset_zeros[index,0]
        index += 1
    return dataset_zeros[6:]

def parameter_opt_for_lasso(data):
    Ci_range = [0.01, 0.5, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100]
    mean_error = []
    std_error = []
    std_error_r2 = []
    mean_parameter_lasso = []
    r2 = []
    X = data[:,:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    for Ci in Ci_range:
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=1/(2 * Ci))
        temp = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            from sklearn.metrics import r2_score
            if abs(r2_score(y[test],ypred)) <= 1: 
                r2.append(r2_score(y[test],ypred))
            else:
                r2.append(0)
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        std_error_r2.append(np.array(r2).std())
        mean_parameter_lasso.append(np.array(r2).mean())
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel("C")
    plt.ylabel("Mean square error")
    plt.title(f"C vs MSE of Lasso Model")
    plt.xlim((0.1, 100))
    plt.show()
    plt.errorbar(Ci_range, mean_parameter_lasso, yerr=std_error_r2)
    plt.xlabel("C")
    plt.ylabel("R2")
    plt.title(f"C vs R2 Score of Lasso Model")
    plt.xlim((0.1, 100))
    plt.show()

def parameter_opt_for_ridge(data):
    Ci_range = [0.01, 0.5, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100]
    mean_error = []
    std_error = []
    std_error_r2 = []
    mean_parameter_ridge = []
    r2 = []
    X = data[:,:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    for Ci in Ci_range:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1/(2 * Ci))
        temp = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            from sklearn.metrics import r2_score
            if abs(r2_score(y[test],ypred)) <= 1: 
                r2.append(r2_score(y[test],ypred))
            else:
                r2.append(0)
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        std_error_r2.append(np.array(r2).std())
        mean_parameter_ridge.append(np.array(r2).mean())
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel("C")
    plt.ylabel("Mean square error")
    plt.title(f"C vs MSE of Ridge Model")
    plt.xlim((0.1, 100))
    plt.show()
    plt.errorbar(Ci_range, mean_parameter_ridge, yerr=std_error_r2)
    plt.xlabel("C")
    plt.ylabel("R2")
    plt.title(f"C vs R2 Score of Ridge Model")
    plt.xlim((0.1, 100))
    plt.show()
    plt.errorbar(Ci_range[:6], mean_parameter_ridge[:6], yerr=std_error_r2[:6])
    plt.xlabel("C")
    plt.ylabel("R2")
    plt.title(f"C vs R2 Score of Ridge Model")
    plt.xlim((0.1, Ci_range[6]))
    plt.show()

def parameter_opt_for_neural_net(data):
    Ci_range = [1, 5, 10, 25, 50, 75, 100]
    mean_error = []
    std_error = []
    std_error_r2 = []
    mean_parameter_nn = []
    r2 = []
    X = data[:,:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    for Ci in Ci_range:
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(alpha=1/(2 * Ci),early_stopping=True)
        temp = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            from sklearn.metrics import r2_score
            if abs(r2_score(y[test],ypred)) <= 1: 
                r2.append(r2_score(y[test],ypred))
            else:
                r2.append(0)
            print(Ci)
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        std_error_r2.append(np.array(r2).std())
        mean_parameter_nn.append(np.array(r2).mean())
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel("C")
    plt.ylabel("Mean square error")
    plt.title(f"C vs MSE of NN Model")
    plt.xlim((0.1, 100))
    plt.show()
    plt.errorbar(Ci_range, mean_parameter_nn, yerr=std_error_r2)
    plt.xlabel("C")
    plt.ylabel("R2")
    plt.title(f"C vs R2 Score of NN Model")
    plt.xlim((0.1, 100))
    plt.show()

def parameter_opt_for_knn(data):
    N_range = [1, 3, 5, 7, 9, 11, 13,int(math.sqrt(data.shape[0]))]
    mean_error = []
    std_error = []
    std_error_r2 = []
    mean_parameter_nn = []
    r2 = []
    X = data[:,:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    for N in N_range:
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=N)
        temp = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            from sklearn.metrics import r2_score
            if abs(r2_score(y[test],ypred)) <= 1: 
                r2.append(r2_score(y[test],ypred))
            else:
                r2.append(0)
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        std_error_r2.append(np.array(r2).std())
        mean_parameter_nn.append(np.array(r2).mean())
    plt.errorbar(N_range, mean_error, yerr=std_error)
    plt.xlabel("C")
    plt.ylabel("Mean square error")
    plt.title(f"C vs MSE of KNN Model")
    plt.xlim((0.1, N_range[len(N_range)-1]))
    plt.show()
    plt.errorbar(N_range, mean_parameter_nn, yerr=std_error_r2)
    plt.xlabel("C")
    plt.ylabel("R2")
    plt.title(f"C vs R2 Score of KNN Model")
    plt.xlim((0.1, N_range[len(N_range)-1]))
    plt.show()

def baseline_1(data,t):
    r =  (data[:,0] - data[:,data.shape[1]-8])/6
    r_index= 0
    while r_index < r.shape[0]:
        if r[r_index] > 2:
            r[r_index] = 0
        r_index += 1
    return data[:,0] + r * t

def baseline_2(data,_):
    return np.ones(data.shape[0], dtype=int) * np.mean(data[:,0])

if main():
    #reproduction_rate_graph()
    #X = concatenate_data()
    #X = make_n_day_prediction_dataset(X,14)
    #parameter_opt_for_lasso(X)
    ##lasso C = [25-75] on MSE, amd [75+] on R2
    ##C = 75
    #parameter_opt_for_ridge(X)
    ##ridge C = [25-] on MSE, amd [10-] on R2
    ##C= 10
    #parameter_opt_for_neural_net(X)
    ##NN C = [25-] on MSE, amd [10-] on R2
    #parameter_opt_for_knn(X)
    ##N > 2 on MSE, and N < 5 on R2
    ##N=2

    #X = concatenate_data()
    #X = make_n_day_prediction_dataset(X,10)
    #parameter_opt_for_lasso(X)
    ##lasso C = [50-] on MSE, amd [75+] on R2
    ##C = 50
    #parameter_opt_for_ridge(X)
    ##ridge C = [25-] on MSE, amd [10-] on R2
    ##C= 25
    ##parameter_opt_for_neural_net(X)
    ##NN C = [25-] on MSE, amd [10-] on R2
    #parameter_opt_for_knn(X)
    ##N > 2 on MSE, and N < 5 on R2
    ##N=5

    #X = concatenate_data()
    #X = make_n_day_prediction_dataset(X,7)
    #parameter_opt_for_lasso(X)
    ##lasso C = [10+] on MSE, amd [75+] on R2
    ##C = 100
    #parameter_opt_for_ridge(X)
    ##ridge C = [any] on MSE, and [10+] on R2
    ##C= 25
    #parameter_opt_for_neural_net(X)
    ##NN C = [25-] on MSE, amd [10-] on R2
    #parameter_opt_for_knn(X)
    ##N > 2 on MSE, and N < 5 on R2
    ##N = 5

    #X = concatenate_data()
    #X = make_n_day_prediction_dataset(X,3)
    #parameter_opt_for_lasso(X)
    ##lasso C = [5-25] on MSE, amd [75+] on R2
    ##C = 75
    #parameter_opt_for_ridge(X)
    ##ridge C = [any] on MSE, amd [5-] on R2
    ##C= 5
    #parameter_opt_for_neural_net(X)
    ##NN C = [25-] on MSE, amd [10-] on R2
    #parameter_opt_for_knn(X)
    ##N > 2 on MSE, and N < 5 on R2
    ##N = 23

    X = concatenate_data()
    pos_data = X[:,6]
    neg_data = X[:,7]
    days = np.arange(X.shape[0])
    plt.plot(days, neg_data,'ro--',label="Compound Negative Sentiment",linewidth=1.5,alpha=0.5) 
    plt.plot(days, pos_data,'go--',label="Compound Positive Sentiment",linewidth=1.5,alpha=0.5) 
    plt.xlabel("day")
    plt.ylabel("Compound Sentiment")
    plt.title(f"Compound Sentiment across Days")
    plt.xlim((0, X.shape[0]))
    plt.legend()
    plt.show()
    days_ahead = [3,7,10,14]
    c_lasso = [75,100,50,75]
    c_ridge = [10,25,25,10]
    n_knn = [23,5,5,3]
    lasso_mse = []
    ridge_mse = []
    knn_mse = []
    b1_mse = []
    b2_mse = []
    lasso_r2 = []
    ridge_r2 = []
    knn_r2 = []
    b1_r2 = []
    b2_r2 = []
    lasso_coefficients = []
    ridge_coefficients = []
    knn_coefficients = []
    for index in [0,1,2,3]:
        X = concatenate_data()
        X = make_n_day_prediction_dataset(X,days_ahead[index])
        X_only = X[:,:X.shape[1]-1]
        y_only = X[:,X.shape[1]-1]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_only, y_only, test_size=0.2,random_state=3)
        from sklearn.linear_model import Lasso
        lasso_model = Lasso(alpha=1/(2 * c_lasso[index]))
        from sklearn.linear_model import Ridge
        ridge_model = Ridge(alpha=1/(2 * c_ridge[index]))
        from sklearn.neighbors import KNeighborsRegressor
        knn_model = KNeighborsRegressor(n_neighbors=n_knn[index])
        lasso_model.fit(X_train, y_train)
        ridge_model.fit(X_train, y_train)
        knn_model.fit(X_train, y_train)
        lasso_ypred = lasso_model.predict(X_test)
        ridge_ypred = ridge_model.predict(X_test)
        knn_ypred = knn_model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        lasso_mse.append(mean_squared_error(y_test, lasso_ypred))
        from sklearn.metrics import r2_score
        lasso_r2.append(r2_score(y_test, lasso_ypred))
        ridge_mse.append(mean_squared_error(y_test, ridge_ypred))
        ridge_r2.append(r2_score(y_test, ridge_ypred))
        knn_mse.append(mean_squared_error(y_test, knn_ypred))
        knn_r2.append(r2_score(y_test, knn_ypred))
        b1_mse.append(mean_squared_error(y_test, baseline_1(X_test,days_ahead[index])))
        b1_r2.append(r2_score(y_test, baseline_1(X_test,days_ahead[index])))
        b2_mse.append(mean_squared_error(y_test, baseline_2(X_test,days_ahead[index])))
        b2_r2.append(r2_score(y_test, baseline_2(X_test,days_ahead[index])))
        lasso_coefficients.append(np.transpose(lasso_model.coef_))
        ridge_coefficients.append(np.transpose(ridge_model.coef_))
    
    for i in range(len(lasso_coefficients)):
        plt.plot(np.arange(lasso_coefficients[i].shape[0]), lasso_coefficients[i],color="blue",label="lasso parameters") 
        plt.xlabel("Parameter")
        plt.ylabel("Value")
        plt.title(f"Parameter Values")
        plt.legend()
        plt.show()
        plt.plot(np.arange(lasso_coefficients[i].shape[0]), ridge_coefficients[i],color="green",label="ridge parameters") 
        plt.xlabel("Parameter")
        plt.ylabel("Value")
        plt.title(f"Parameter Values")
        plt.legend()
        plt.show() 
        
    print("MSE:")
    print(f"day {days_ahead}")
    print(f"bl1 {b1_mse}")
    print(f"bl2 {b2_mse}")
    print(f"las {lasso_mse}")
    print(f"rdg {ridge_mse}")
    print(f"knn {knn_mse}")
    print()
    print("R_2 Score:")
    print(f"day {days_ahead}")
    print(f"bl1 {b1_r2}")
    print(f"bl2 {b2_r2}")
    print(f"las {lasso_r2}")
    print(f"rdg {ridge_r2}")
    print(f"knn {knn_r2}")
    plt.plot(days_ahead, b1_r2,color="blue", marker='o',label="exponential baseline") 
    plt.plot(days_ahead, b2_r2,color="green", marker='o',label="mean baseline") 
    plt.plot(days_ahead, lasso_r2,color="red", marker='o',label="lasso regression") 
    plt.plot(days_ahead, ridge_r2,color="orange", marker='o',label="ridge regression") 
    plt.plot(days_ahead, knn_r2,color="purple", marker='o',label="knn regression") 
    plt.xlabel("n, days into future prediction")
    plt.ylabel("R^2 Score")
    plt.title(f"Model Predictions for day n versus R^2 Score")
    plt.xlim((3, days_ahead[len(days_ahead)-1]))
    plt.legend()
    plt.show()
    plt.plot(days_ahead, b1_mse,color="blue", marker='o',label="exponential baseline") 
    plt.plot(days_ahead, b2_mse,color="green", marker='o',label="mean baseline") 
    plt.plot(days_ahead, lasso_mse,color="red", marker='o',label="lasso regression") 
    plt.plot(days_ahead, ridge_mse,color="orange", marker='o',label="ridge regression") 
    plt.plot(days_ahead, knn_mse,color="purple", marker='o',label="knn regression") 
    plt.xlabel("n, days into future prediction")
    plt.ylabel("MSE")
    plt.title(f"Model Predictions for day n versus MSE")
    plt.xlim((3, days_ahead[len(days_ahead)-1]))
    plt.legend()
    plt.show()


