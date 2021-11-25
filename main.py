import json
from collections import Counter

import nltk
import plotly.graph_objects as go
from nltk.corpus import stopwords

import tweet_analyzer
import tweet_preprocesser
from tweet_analyzer import TweetAnalyzer as TA
from tweet_preprocesser import TweetPreprocessing as TP

# tweet_analyzer.analyze('tweet_from_2016_to_2020.json')


def preprocess_all():
    print('\n --- \n')
    mf_2016 = tweet_preprocesser.preprocess(
        'milano_finanza/milano_finanza_2016-01-01T00_00_00Z_2016-12-31T23_59_59Z.json', 'JSON/milano_finanza_2016_sumup')
    print('\n --- \n')
    mf_2017 = tweet_preprocesser.preprocess(
        'milano_finanza/milano_finanza_2017-01-01T00_00_00Z_2017-12-31T23_59_59Z.json', 'JSON/milano_finanza_2017_sumup')
    print('\n --- \n')
    mf_2018 = tweet_preprocesser.preprocess(
        'milano_finanza/milano_finanza_2018-01-01T00_00_00Z_2018-12-31T23_59_59Z.json', 'JSON/milano_finanza_2018_sumup')
    print('\n --- \n')
    mf_2019 = tweet_preprocesser.preprocess(
        'milano_finanza/milano_finanza_2019-01-01T00_00_00Z_2019-12-31T23_59_59Z.json', 'JSON/milano_finanza_2019_sumup')
    print('\n --- \n')
    mf_2020 = tweet_preprocesser.preprocess(
        'milano_finanza/milano_finanza_2020-01-01T00_00_00Z_2020-12-31T23_59_59Z.json', 'JSON/milano_finanza_2020_sumup')
    print('\n --- \n')
    print('--- \n')


def analyze_all():
    print('\n> 2016')
    analyzer = TA(open('milano_finanza_2016_sumup.json'))
    analyzer.get_tweet_statistics()
    print('\n> 2017')
    analyzer = TA(open('milano_finanza_2017_sumup.json'))
    analyzer.get_tweet_statistics()
    print('\n> 2018')
    analyzer = TA(open('milano_finanza_2018_sumup.json'))
    analyzer.get_tweet_statistics()
    print('\n> 2019')
    analyzer = TA(open('milano_finanza_2019_sumup.json'))
    analyzer.get_tweet_statistics()
    print('\n> 2020')
    analyzer = TA(open('milano_finanza_2020_sumup.json'))
    analyzer.get_tweet_statistics()


def wrap_tweet(filenames, outname):
    tweet_total = []

    for filename in filenames:
        for tweet in json.load(open(filename)):
            tweet_total.append(tweet)

    with open('{}.json'.format(outname), 'w') as outfile:
        json.dump(tweet_total, outfile)


def search_correlate_tokens(tweet_json, word):
    correlate = []
    for tweet in tweet_json:
        if word in tweet['tokenized_text']:
            for token in tweet['tokenized_text']:
                correlate.append(token)
    correlate = Counter(correlate)
    return {k: v for k, v in sorted(
        correlate.items(), key=lambda item: item[1], reverse=True)}


def search_correlate_entities(tweet_json, word, tool):
    correlate = []
    for tweet in tweet_json:
        if word in tweet['tokenized_text']:
            if tool == 'spacy':
                if tweet['entity_labels'] != []:
                    for entity in tweet['entity_labels']:
                        correlate.append(entity[1])
            if tool == 'context_annotations':
                if tweet['context_annotations'] != []:
                    for entity in tweet['context_annotations']:
                        correlate.append(entity[1])
            if tool == 'entities_annotations':
                if tweet['entities_annotations'] != []:
                    for entity in tweet['entities_annotations']:
                        correlate.append(entity[1])
    corr_dict = Counter(correlate)
    return {k: v for k, v in sorted(
        corr_dict.items(), key=lambda item: item[1], reverse=True)}, set(corr_dict)

# wrap_tweet(['JSON/milano_finanza_2016_sumup.json', 'JSON/milano_finanza_2017_sumup.json', 'JSON/milano_finanza_2018_sumup.json','JSON/milano_finanza_2019_sumup.json', 'JSON/milano_finanza_2020_sumup.json'], 'tweet_from_2016_to_2020')


analizer = TA(open('tweet_from_2016_to_2020.json'))
# analizer.print_trends_graph('prova', [['brexit'], ['piazza affari'], ['borsa'], ['brexit', 'accordo'], ['brexit', 'ue'], ['brexit', 'sterlina'], ['brexit', 'may'], ['brexit', 'dopo'], ['brexit', 'no'], ['brexit', 'borse'], ['brexit', 'piazza affari'], ['brexit', 'hard']])
ent_dict, ent_set = search_correlate_entities(
    analizer.data_full, 'brexit', 'spacy')
analizer.print_line_graph(ent_dict, 'prova ent')
