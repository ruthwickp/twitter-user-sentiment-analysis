from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pprint import pprint
from tweet_dumper import *
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from collections import Counter
import os.path


matplotlib.style.use('ggplot')

MIN_TWEET_THRESHOLD = 10

MONTH_TIMEFRAME = 'month'
WEEK_TIMEFRAME = 'week'


class SentimentAggregator:
    def __init__(self, timeframe, compound_scores, timestamps):
        self.timeframe = timeframe
        self.compound_scores = compound_scores
        self.timestamps = timestamps
        self.aggregate_info = None
        self.aggregate_by_timeframe()

    def aggregate_by_timeframe(self):
        data_dict = {}
        for i, x in enumerate(self.timestamps):
            key_timeframe = self.generate_key_from_date_with_timeframe(x)
            sentiment_type = self.compute_sentiment_type(self.compound_scores[i])

            if key_timeframe not in data_dict:
                data_dict[key_timeframe] = []
            data_dict[key_timeframe].append(sentiment_type)


        self.aggregate_info = {}
        for k, v in data_dict.iteritems():
            if len(v) < MIN_TWEET_THRESHOLD:
                continue

            c = Counter(v)
            percentages = [100 * x / float(len(v)) for x in [c[1], c[0], c[-1]]]
            self.aggregate_info[k] = percentages


    def compute_sentiment_type(self, num):
        if num >= .5:
            return 1
        if num <= -.5:
            return -1
        return 0

    def generate_key_from_date_with_timeframe(self, dt):
        if self.timeframe == MONTH_TIMEFRAME:
            return dt.strftime("%Y-%m")
        elif self.timeframe == WEEK_TIMEFRAME:
            return self.next_weekday(dt, 0).strftime("%Y-%m-%d")


    def get_aggregate_info(self):
        assert self.aggregate_info != None
        return self.aggregate_info

    def next_weekday(self, d, weekday):
        days_ahead = weekday - d.weekday()
        if days_ahead <= 0: # Target day already happened this week
            days_ahead += 7
        return d + timedelta(days_ahead)

        

class SentimentAnalyzer:
    def __init__(self, twitter_name):
        self.twitter_name = twitter_name
        self.analyzer = SentimentIntensityAnalyzer()

        self.data = None
        self.twitter_csv_file = None
        self.load_data()

    def load_data(self):
        self.twitter_csv_file = self.twitter_name + '_tweets.csv'
        if not os.path.isfile(self.twitter_csv_file):
            get_all_tweets(self.twitter_name)
        self.load_tweet_data_from_csv()

    def load_tweet_data_from_csv(self):
        self.data = []
        with open(self.twitter_csv_file, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.data.append(row)
                assert len(row) == 3
        self.data = np.array(self.data)


    def get_sentiment_by_timeframe(self, timeframe):
        sentiment_scores_dict = self.analyze_sentences(self.data[:,2][1:])
        timestamps = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in self.data[:,1][1:]]
        compound_scores = [x['compound'] for x in sentiment_scores_dict]

        aggregate_info = SentimentAggregator(timeframe, compound_scores, 
            timestamps).get_aggregate_info()
        df = pd.DataFrame.from_dict(aggregate_info, orient='index')
        df.columns = ['Positive', 'Neutral', 'Negative']
        df = df.reindex(index=sorted(df.index))
        return df

    def analyze_sentences(self, sentences):
        scores = []
        for sentence in sentences:
            vs = self.analyzer.polarity_scores(sentence)
            scores.append(vs)
        return scores


if __name__ == '__main__':
    ### Example twitter handles (replace with your own)
    for df_twitter_name in ['realDonaldTrump', 'KellyannePolls', 'IvankaTrump']:
        sentiment = SentimentAnalyzer(df_twitter_name)
        
        ### Choose timeframe below (WEEK_TIMEFRAME, MONTH_TIMEFRAME)
        df = sentiment.get_sentiment_by_timeframe(WEEK_TIMEFRAME)
        y_parameters = ['Positive', 'Negative']
        ax = df.plot.bar(x=df.index.name, y=y_parameters, colormap='Paired', title=df_twitter_name)
    plt.show()  
