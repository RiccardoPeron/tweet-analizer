import json

import explorer as TE
import preprocesser as TP


def check_wrap(filename):
    f = json.load(open(filename))
    ids = set()
    for t in range(len(f)):
        ids.add(f[t]["id"])
    if len(ids) != len(f):
        print(" >>> ERROR <<< ")


# analizer = TA(open("tweet_from_2016_to_2020.json"))
# analizer.print_trends_graph('prova', [['brexit'], ['piazza affari'], ['borsa'], ['brexit', 'accordo'], ['brexit', 'ue'], ['brexit', 'sterlina'], ['brexit', 'may'], ['brexit', 'dopo'], ['brexit', 'no'], ['brexit', 'borse'], ['brexit', 'piazza affari'], ['brexit', 'hard']])

merged_file_name = TP.preprocess("sources_en_journal", "en")
TE.explore2(
    merged_file_name,
    # "merged_tweet_sources_en.json",
    "en",
    [
        "covid-19",
        "covid",
        "covid 19",
        "coronavirus",
        "corona virus",
    ],
    day_span=15,
)
