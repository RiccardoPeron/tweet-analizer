import json
from collections import Counter
import os

import nltk
import plotly.graph_objects as go
from nltk.corpus import stopwords

from tqdm import tqdm

# import tweet_analyzer
# import tweet_preprocesser
# from tweet_analyzer import TweetAnalyzer as TA
# from tweet_preprocesser import TweetPreprocessing as TP

# tweet_analyzer.analyze('tweet_from_2016_to_2020.json')


def wrap_tweet(outname):
    tweet_total = []

    path = os.getcwd() + "/JSON/"
    for dir in os.listdir(path):
        filedir = path + "/" + dir
        files = []
        lengths = []
        index = []
        file_number = 0
        for file in os.listdir(filedir):
            f = json.load(open(filedir + "/" + file))
            files.append(f)
            lengths.append(len(f))
            index.append(0)
            file_number += 1

        tot_tweets = sum(lengths)
        for i in tqdm(range(tot_tweets)):
            tweets = []
            for j in range(file_number):
                if index[j] < lengths[j]:
                    tweets.append(files[j][index[j]]["created_at"])
                else:
                    tweets.append("9999-12-31T09:12:16+00:00")

            idx = tweets.index(min(tweets))
            tweet_total.append(files[idx][index[idx]])
            index[idx] += 1

    with open("{}.json".format(outname), "w") as outfile:
        json.dump(tweet_total, outfile)


def check_wrap(filename):
    f = json.load(open(filename))
    ids = set()
    for t in range(len(f)):
        ids.add(f[t]["id"])
    if len(ids) != len(f):
        print(" >>> ERROR <<< ")


wrap_tweet("tweet_from_2016_to_2020")
# check_wrap("tweet_from_2016_to_2020.json")


# analizer = TA(open("tweet_from_2016_to_2020.json"))
# analizer.print_trends_graph('prova', [['brexit'], ['piazza affari'], ['borsa'], ['brexit', 'accordo'], ['brexit', 'ue'], ['brexit', 'sterlina'], ['brexit', 'may'], ['brexit', 'dopo'], ['brexit', 'no'], ['brexit', 'borse'], ['brexit', 'piazza affari'], ['brexit', 'hard']])
