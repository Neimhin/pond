import string
import nltk
import numpy as np
import pandas as pd
import re
import os
import sys
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def setup():
    nltk.download([
     "names",
     "stopwords",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt",
 ])

def main():
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()
    data = list(set([f.replace(".csv","") for f in os.listdir("./pushshift") if ".csv" in f]))
    sentiment = pd.DataFrame()
    text_data = pd.DataFrame()
    count = 0
    count_t = 0
    for day in data:
        count_t = 0
        all_text = ""
        progress(count, len(data))
        #max_sentiment = [0,0,0]
        max_sentiment =  {'neg': 0, 'pos': 0}
        #day_score_sent = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0, "total":0}
        day_score_sent = {'p_neg': 0.0, 'p_pos': 0.0}
        try:
            daily_data = pd.read_csv(f"./pushshift/{day}.csv",usecols=["body"])
            for _,r in daily_data.iterrows():
                if re.search("\* \*\*",r["body"]) or re.search("\*I bot,",r["body"]):
                    continue
                text = ""
                for word in r["body"].split(" "):
                    if word.lower() not in stop_words:
                        if re.search("https",word) == None and re.search("&gt;",word) == None and re.search("tldr;",word) == None and re.search("/r/;",word) == None:
                            if text == "":
                                text += word
                            else:
                                text += f" {word}"
                text = text.replace("[deleted]","").replace("[removed]","").replace("\*Spoiler\*","").replace("  "," ")
                count_t += 1
                #sent_score = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
                sent_score = {'p_neg': 0.0,'p_pos': 0.0}
                sentence_no = 0
                all_text += " " + text
                for t in nltk.sent_tokenize(text):
                #    text_score = sia.polarity_scores(t)
                    sentence_no += 1
                #    for k in sent_score.keys():
                #        sent_score[k] += text_score[k]
                    blob = TextBlob(t)
                    (p_pos,p_neg) = blob.sentiment
                    sent_score['p_neg'] += p_neg
                    sent_score['p_pos'] +=p_pos
                    if p_neg > p_pos:
                        max_sentiment["neg"] += 1
                    else:
                        max_sentiment["pos"] += 1
                #for k in sent_score.keys():
                #    if sentence_no > 0:
                #        day_score_sent[k] += sent_score[k]/sentence_no
                #    else:
                #        day_score_sent[k] += sent_score[k]
                if sentence_no > 0:
                    day_score_sent['p_neg'] += sent_score['p_neg']/ sentence_no
                    day_score_sent['p_pos'] +=sent_score['p_pos']/ sentence_no
                    max_sentiment["pos"] = max_sentiment["pos"]/ sentence_no
                    max_sentiment["neg"] = max_sentiment["neg"]/ sentence_no
                #pos_neg_neu = [day_score_sent["pos"],day_score_sent["neg"],day_score_sent["neu"]]
                #max_sentiment[pos_neg_neu.index(max(pos_neg_neu))] += 1
        except:
            pass
        pos = 0
        neg = 0
        #neu = 0
        count_t = count_t
        for k in day_score_sent.keys():
            if count_t != 0:
                if k == "total":
                    pass
                day_score_sent[k] = day_score_sent[k]/count_t
        if count_t != 0:
            #pos = np.log(max_sentiment[0]/count_t)
            #neg = np.log(max_sentiment[1]/count_t)
            #neu = np.log(max_sentiment[2]/count_t)
            if max_sentiment["pos"] != 0:
                pos = np.log(max_sentiment["pos"])
            if max_sentiment["neg"] != 0:
                neg = np.log(max_sentiment["neg"])
            count_t = np.log(count_t)
            scale = 1/ (day_score_sent["p_pos"] + day_score_sent["p_neg"])
            day_score_sent["p_pos"] *= scale
            day_score_sent["p_neg"] *= scale
        #sentiment = sentiment.append({"date":day,"d_s_pos":day_score_sent["pos"],"d_s_neg":day_score_sent["neg"],"d_s_neu":day_score_sent["neu"],"d_s_com":day_score_sent["compound"],"total":count_t, "por_pos":pos,"por_neg":neg,"por_neu":neu},ignore_index=True)
        sentiment = sentiment.append({"date":day,"d_s_pos":day_score_sent["p_pos"],"d_s_neg":day_score_sent["p_neg"],"total":count_t, "por_pos":pos,"por_neg":neg},ignore_index=True)
        text_data = text_data.append({"date":day,"text":all_text},ignore_index=True)
        count += 1
        print(day,day_score_sent["p_pos"],day_score_sent["p_neg"],count_t,pos,neg)
    sentiment.to_csv("./sentiment_data_.csv",columns=["date","d_s_pos","d_s_neg","total","por_pos","por_neg"])
    text_data.to_csv("./text_data_.csv",columns=["date","text"])
if __name__ == "__main__":
    #setup()
    main()