import json
from collections import Counter
from datetime import datetime

import nltk
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from nltk.corpus import stopwords
from nltk.corpus.reader import categorized_sents


class TweetAnalyzer:
    def __init__(self, file, first="", last=""):
        self.data_full = json.load(file)
        self.data_full = self.data_full["tweets"]
        self.data_full = self.delete_stopwords()
        self.total_tokens = 0
        self.total_tweets = 0
        self.vocabulary_size = 0
        if first == "":
            self.__start = datetime.fromisoformat(
                min([tweet["created_at"] for tweet in self.data_full])
            ).replace(tzinfo=None)
        else:
            self.__start = datetime.fromisoformat(first).replace(tzinfo=None)
        if last == "":
            self.__end = datetime.fromisoformat(
                max([tweet["created_at"] for tweet in self.data_full])
            ).replace(tzinfo=None)
        else:
            self.__end = datetime.fromisoformat(last).replace(tzinfo=None)

    def delete_stopwords(self):
        """
        delete all the stopwords that are in a list of words

        Parameters
        ----------
        """
        # stopwords = json.load(open('italian_stopwords.json'))
        nltk.download("stopwords")
        stop_words = set(stopwords.words("italian"))
        stop_words.add("l'")
        stop_words.add("l’")

        for tweet in self.data_full:
            tokens_no_stop = []
            for token in tweet["tokenized_text"]:
                if token not in stop_words:
                    tokens_no_stop.append(token)
                tweet["tokenized_text"] = tokens_no_stop
        return self.data_full

    def get_spacy_entities_labels(self):
        """
        analyze the tweet object and make the list of all the entities subdivided in the 4 catgories (LOC, PER, ORG, MISC)
        and returns the dictionary of each category with the entity assocuated with its number of occurrences

        Parameters
        ----------
        """
        labels = {"LOC": 0, "PER": 0, "ORG": 0, "MISC": 0}

        all = 0
        tweets = 0

        for tweet in self.data_full:
            twt = all
            all += (
                len(tweet["entity_labels"]["LOC"])
                + len(tweet["entity_labels"]["PER"])
                + len(tweet["entity_labels"]["ORG"])
                + len(tweet["entity_labels"]["MISC"])
            )
            labels["LOC"] += len(tweet["entity_labels"]["LOC"])
            labels["ORG"] += len(tweet["entity_labels"]["ORG"])
            labels["PER"] += len(tweet["entity_labels"]["PER"])
            labels["MISC"] += len(tweet["entity_labels"]["MISC"])

            if all > twt:
                tweets += 1

        print("ENTITY LABELS")
        print(
            "\t numero totale di entity labels: {} - {:.4}%".format(
                all, all / self.total_tokens * 100
            )
        )
        print(
            "\t numero totale di entity labels non ripetute nello stesso tweet: {}".format(
                sum(list(labels.values()))
            )
        )
        print(
            "\t numero di tweet che hanno almeno una entity label: {} - {:.4}%".format(
                tweets, tweets / self.total_tweets * 100
            )
        )
        self.total_tweets = tweets
        self.total_tokens = all
        return self.sort_dict(labels)

    def get_spacy_labels_stats(self, label):
        all = 0
        tweets = 0
        categories_stats = []

        for tweet in self.data_full:
            tweet_labels = tweet["entity_labels"][label]

            if (len(tweet_labels)) > 0:
                tweets += 1
                for entity in tweet_labels:
                    categories_stats.append(entity)

        categories_stats = self.get_sorted_dictionary(categories_stats)
        print(
            "\t numero totale di entity labels con label {}: {} - {: .4} %".format(
                label, all, all / self.total_tokens * 100
            )
        )

        print(
            "\t numero totale di entity labels con label {} non ripetute nello stesso tweet: {}".format(
                label, sum(list(categories_stats.values()))
            )
        )
        print(
            "\t numero di tweet che hanno almeno una entity label con label {}: {} - {: .4} %".format(
                label, tweets, tweets / self.total_tweets * 100
            )
        )
        return categories_stats

    def get_document_context_annotations(self):
        context_annotations = []

        # for tweet in self.data_full:
        #     for touple in tweet['context_annotations']:
        #         context_annotations.append(touple[0])

        for tweet in self.data_full:
            for touple in tweet["context_annotations"]:
                # if touple[0] == 'Entities [Entity Service]':
                context_annotations.append(touple[0])

        dict_ = self.get_sorted_dictionary(context_annotations)

        return set(context_annotations)

    def get_context_annotations_labels(self):
        """
        analyzes the tweet object and return the dictionary of the labels for each entity_annotations field associated with the number of tweets which the entity is in it

        Parameters
        ----------
        """
        classes = json.load(open("utilities/context_annotation_labels.json"))
        labels_list = list(classes.keys())
        labels = Counter(labels_list)
        labels = dict.fromkeys(labels, 0)
        all = 0
        tweets = 0

        for tweet in self.data_full:
            twt = all
            for label in labels_list:
                all += len(tweet["context_annotations"][label])
                labels[label] += len(tweet["context_annotations"][label])
            if all > twt:
                tweets += 1

        print("CONTEXT ANNOTATIONS")
        print(
            "\t numero totale di context annotations: {} - {:.4}%".format(
                all, all / self.total_tokens * 100
            )
        )
        print(
            "\t numero totale di context annotations non ripetute nello stesso tweet: {}".format(
                sum(list(labels.values()))
            )
        )
        print(
            "\t numero di tweet che hanno almeno una context annotation: {} - {:.4}%".format(
                tweets, tweets / self.total_tweets * 100
            )
        )
        self.total_tweets = tweets
        self.total_tokens = all
        return self.sort_dict(labels)

    def get_context_label_stats(self, label):
        """
        analyzes the tweet object and return the dictionary of the labels of the specified entity_annotations field associated with the number of tweets which the entity is in it

        Parameters
        ----------
        label: string
        file: json file
            file that contains the context annotations lables
        """
        all = 0
        tweets = 0
        categories_stats = []

        for tweet in self.data_full:
            tweet_labels = tweet["context_annotations"][label]

            if len(tweet_labels) > 0:
                tweets += 1
                for entity in tweet_labels:
                    categories_stats.append(entity)

        categories_stats = self.get_sorted_dictionary(categories_stats)
        print(
            "\t numero totale di context annotations con label {}: {} - {: .4} %".format(
                label, all, all / self.total_tokens * 100
            )
        )
        print(
            "\t numero totale di context annotations con label {} non ripetute nello stesso tweet: {}".format(
                label, sum(list(categories_stats.values()))
            )
        )
        print(
            "\t numero di tweet che hanno almeno una context annotation con label {}: {} - {: .4} %".format(
                label, tweets, tweets / self.total_tweets * 100
            )
        )
        return categories_stats

    def get_entities_annotations_labels(self):
        labels_list = ["Place", "Product", "Person", "Organization", "Other"]
        labels = Counter(labels_list)
        labels = dict.fromkeys(labels, 0)

        all = 0
        tweets = 0

        for tweet in self.data_full:
            twt = all
            for label in labels_list:
                all += len(tweet["entities_annotations"][label])
                labels[label] += len(tweet["entities_annotations"][label])
            if all > twt:
                tweets += 1

        print("ENTITIES ANNOTATIONS")
        print(
            "\t numero totale di context annotations: {} - {:.4}%".format(
                all, all / self.total_tokens * 100
            )
        )
        print(
            "\t numero totale di context annotations non ripetute nello stesso tweet: {}".format(
                sum(list(labels.values()))
            )
        )
        print(
            "\t numero di tweet che hanno almeno una context annotation: {} - {:.4}%".format(
                tweets, tweets / self.total_tweets * 100
            )
        )
        self.total_tweets = tweets
        self.total_tokens = all
        return self.sort_dict(labels)

    def get_entities_label_stats(self, label):
        all = 0
        tweets = 0
        categories_stats = []

        for tweet in self.data_full:
            tweet_labels = tweet["entities_annotations"][label]

            if len(tweet_labels) > 0:
                tweets += 1
                for entity in tweet_labels:
                    categories_stats.append(entity)

        categories_stats = self.get_sorted_dictionary(categories_stats)
        print(
            "\t numero totale di context annotations con label {}: {} - {: .4} %".format(
                label, all, all / self.total_tokens * 100
            )
        )
        print(
            "\t numero totale di context annotations con label {} non ripetute nello stesso tweet: {}".format(
                label, sum(list(categories_stats.values()))
            )
        )
        print(
            "\t numero di tweet che hanno almeno una context annotation con label {}: {} - {: .4} %".format(
                label, tweets, tweets / self.total_tweets * 100
            )
        )
        return categories_stats

    def sort_dict(self, myDict, reverse_=True):
        """
        given a dictionary returns the sorted dictionary

        Parameters
        ----------
        myDict: dict
        reverse_: bool
          - True: ascending order
          - False: descending order
        """
        return {
            k: v
            for k, v in sorted(
                myDict.items(), key=lambda item: item[1], reverse=reverse_
            )
        }

    def get_sorted_dictionary(self, list_):
        """
        given a list returns the sorted dictionary with the elements associated with the number of occurrences

        Parameters
        ----------
        list_ : list
        """
        dict_ = self.sort_dict(Counter(list_))

        return dict_

    def field_compatter(self, field, start, end):
        """
        given a field name returns a list of all the values of this field for all the tweets in a time span

        Parameters
        ----------
        field: string
          name of the field
        start: datetime object
          date of the first tweet to analyze
        end: datetime object
          date of the last tweet to analyze
        """
        # if self.start > self.end:
        # raise ValueError('start time can not be greater than end time')
        list_ = []
        for tweet in self.data_full:
            if (
                datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                >= start
                and datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                <= end
            ):
                list_.append(tweet[field])
        return list_

    def unpack_list(self, llist):
        """
        given a list of lisy returns a list that contains all the elements in all the sublists

        Parameters
        ----------
        llist: list of list
        """
        list_ = []
        for item_list in llist:
            for item in item_list:
                list_.append(item)
        return list_

    def get_token_dict(self):
        """
        returns the sorted dictionary of all the tokens associated with its number of occurrences

        Parameters
        ----------
        """
        return self.sort_dict(
            Counter(
                self.unpack_list(
                    self.field_compatter("tokenized_text", self.__start, self.__end)
                )
            )
        )

    def get_tweet_statistics(self):
        """
        prints some statistics of the dataset of tweets such as:
        - total number of tweets
        - total number of tokens
        - number of unique tokens
        - average lenght of a tweet (in tokens)
        - lenght of the longest tweet (in token)

        Parameters
        ----------
        """
        # total tweet number
        self.total_tweets = len(self.data_full)
        print("total tweets: {}".format(self.total_tweets))
        # total token number
        tokenized_text = self.field_compatter(
            "tokenized_text", self.__start, self.__end
        )
        tokens = self.unpack_list(tokenized_text)
        self.total_tokens = len(tokens)
        print("total tokens: {}".format(self.total_tokens))
        # different tokens number
        token_dict = self.sort_dict(Counter(tokens))
        self.vocabulary_size = len(token_dict.keys())
        print("different tokens number: {}".format(self.vocabulary_size))
        # average tweet lenght
        print("average tweet lenght: {0:.4}".format(len(tokens) / len(self.data_full)))
        # max tweet len
        print("max tweet len: {}".format(max([len(list_) for list_ in tokenized_text])))

    def print_line_graph(self, text_dict, imagename, size=50):
        # len(list(text_dict.keys()))
        total_items = sum(list(text_dict.values()))
        values = list(text_dict.values())[0:size]
        values = [val / len(self.data_full) for val in values]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(text_dict.keys())[0:size],
                y=values,
                text=list(text_dict.values()),
                mode="lines+markers",  # +text',
                textposition="top right",
            )
        )
        fig.update_layout(
            title="{} ({} - {})".format(
                imagename, self.__start.strftime("%b %Y"), self.__end.strftime("%b %Y")
            ),
            xaxis_title="Words",
            yaxis_title="Tweets",
            yaxis_tickformat=",.1%",
            # width=max(total_items/2, 1500)
        )
        fig.write_image("./{}.png".format(imagename))
        print("> printed {}".format(imagename))

    def print_trends_graph(self, imagename, entity_list=[], size=5):
        if entity_list == []:
            entity_list = [
                [item] for item in list((self.get_token_dict()).keys())[0:size]
            ]

        entity_stats = {}
        for ent in entity_list:
            entity_stats["/".join(ent)] = {}

        n_months = (self.__end.year - self.__start.year) * 12 + (
            self.__end.month - self.__start.month
        )
        fig = go.Figure()

        for month in range(n_months):
            date_start = self.__start + relativedelta(months=month)
            date_end = self.__start + relativedelta(months=month + 1)

            tweets = self.field_compatter("tokenized_text", date_start, date_end)
            for ent in entity_list:
                entity_stats["/".join(ent)][date_start.strftime("%m %Y")] = 0
                for tweet in tweets:
                    tweet = set(tweet)
                    if len(set(ent) & tweet) == len(set(ent)):
                        entity_stats["/".join(ent)][date_start.strftime("%m %Y")] += 1

        for ent in entity_list:
            val = [
                n / len(self.data_full)
                for n in list(entity_stats["/".join(ent)].values())
            ]
            fig.add_trace(
                go.Scatter(
                    x=list(entity_stats["/".join(ent)].keys()),
                    y=val,
                    text=list(entity_stats["/".join(ent)].values()),
                    mode="lines+markers+text",
                    textposition="middle left",
                    name="/".join(ent),
                )
            )

        fig.update_layout(
            title="{} ({} - {})".format(
                imagename, self.__start.strftime("%b %Y"), self.__end.strftime("%b %Y")
            ),
            xaxis_title="Months",
            yaxis_title="Number of usage",
            # yaxis_nticks=10,
            yaxis_tickformat=",.1%",
            width=n_months * 100,
            height=len(entity_list) * 150,
        )
        fig.write_image("./{}.png".format(imagename))
        print("> printed {}".format(imagename))


def analyze(filename):
    tweet_file = open(filename)
    analyzer = TweetAnalyzer(tweet_file)

    print("> tweets statistic")
    analyzer.get_tweet_statistics()

    analyzer.print_line_graph(
        analyzer.get_token_dict(), "IMG/figure 0: tweet words distribution"
    )
    print("----")
    analyzer.print_line_graph(
        analyzer.get_spacy_entities_labels(), "IMG/figure 1: spacy entity labels"
    )
    analyzer.print_line_graph(
        analyzer.get_spacy_labels_stats("LOC"), "IMG/figure 2: loc entity distribution"
    )
    analyzer.print_line_graph(
        analyzer.get_spacy_labels_stats("ORG"), "IMG/figure 3: org entity distribution"
    )
    analyzer.print_line_graph(
        analyzer.get_spacy_labels_stats("PER"), "IMG/figure 4: per entity distribution"
    )
    analyzer.print_line_graph(
        analyzer.get_spacy_labels_stats("MISC"),
        "IMG/figure 5: misc entity distribution",
    )

    print("----")
    analyzer.print_trends_graph("IMG/figure 6: trends top 5 tokens")
    analyzer.print_trends_graph(
        "IMG/figure 7: trends custom tokens",
        [
            ["ue"],
            ["brexit"],
            ["inghilterra"],
            ["brexit", "ue"],
            ["ue", "gran bretagna"],
            ["ue", "inghilterra"],
            ["inghilterra", "euro"],
        ],
    )
    print("----")
    analyzer.print_line_graph(
        analyzer.get_context_annotations_labels(),
        "IMG/figure 8: context labels distribution",
    )

    analyzer.print_line_graph(
        analyzer.get_context_label_stats("brand"), "IMG/figure 9: brands distribution"
    )
    analyzer.print_line_graph(
        analyzer.get_context_label_stats("person"), "IMG/figure 10: person distribution"
    )
    analyzer.print_line_graph(
        analyzer.get_context_label_stats("entertainment"),
        "IMG/figure 11: entertainment distribution",
    )
    analyzer.print_line_graph(
        analyzer.get_context_label_stats("sport"), "IMG/figure 12: sport distribution"
    )
    analyzer.print_line_graph(
        analyzer.get_context_label_stats("politics"),
        "IMG/figure 13: politics distribution",
    )
    analyzer.print_line_graph(
        analyzer.get_context_label_stats("videogame"),
        "IMG/figure 14: videogame distribution",
    )
    analyzer.print_line_graph(
        analyzer.get_context_label_stats("product"),
        "IMG/figure 15: products distribution",
    )
    analyzer.print_line_graph(
        analyzer.get_context_label_stats("event"), "IMG/figure 16: events distribution"
    )

    print("----")
    analyzer.print_line_graph(
        analyzer.get_entities_annotations_labels(),
        "IMG/figure 17: entities annotations distribution",
    )

    analyzer.print_line_graph(
        analyzer.get_entities_label_stats("Place"),
        "IMG/figure 18: Place entity annotations distribution",
    )
    analyzer.print_line_graph(
        analyzer.get_entities_label_stats("Organization"),
        "IMG/figure 19: Organization entity annotations distribution",
    )
    analyzer.print_line_graph(
        analyzer.get_entities_label_stats("Product"),
        "IMG/figure 20: Product entity annotations distribution",
    )
    analyzer.print_line_graph(
        analyzer.get_entities_label_stats("Person"),
        "IMG/figure 21: Person entity annotations distribution",
    )
    analyzer.print_line_graph(
        analyzer.get_entities_label_stats("Other"),
        "IMG/figure 22: Other entity annotations distribution",
    )


analyze("merged_tweet_sources_it.json")
