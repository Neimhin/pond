import pandas as pd
import numpy as np
if __name__=="__main__":

    df1=pd.read_csv("world_covid_data.csv",usecols=["date","new_cases"])
    df2=pd.read_csv("sentiment_data_.csv",usecols=["date","d_s_pos","d_s_neg","total","por_pos","por_neg"])
    df2["pos_strength"]=df2["d_s_pos"]*df2["por_pos"]
    df2["neg_strength"]=df2["d_s_neg"]*df2["por_neg"]
    print(df1.head())
    print(df2.head())
    df1=df1.merge(df2,how='left',on='date')
    print(df1.head())
    df1=df1.dropna()
    print(df1.head())
    print(df1.shape)
    #df1["date"]=pd.array(pd.DatetimeIndex(df1["date"]).view(np.int64))/1000000000
    #df1["date"]=(df1["date"]-df1["date"].values[0])/(24*60*60)
    print(df1.tail())
    df1.to_csv("collated_covid_sentiment.csv",columns=["date","new_cases","d_s_pos","d_s_neg","total","por_pos","por_neg","pos_strength","neg_strength"], index=True)