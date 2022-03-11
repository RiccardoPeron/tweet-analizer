import json

import explorer as TE
import preprocesser as TP

# merged_file_name = TP.preprocess("sources_en_journal", "en")
TE.explore2(
    # merged_file_name,
    "merged_tweet_sources_it.json",
    "it",
    [
        "covid-19",
        "covid",
        "covid 19",
        "covid19",
        "coronavirus",
        "corona virus",
    ],
    start="2020-02-01",
    end="2020-03-30",
    day_span=5,
)
TE.test(
    "merged_tweet_sources_it.json",
    "it",
    [
        "covid-19",
        "covid",
        "covid 19",
        "covid19",
        "coronavirus",
        "corona virus",
    ],
    start="2020-02-01",
    end="2020-03-30",
)
