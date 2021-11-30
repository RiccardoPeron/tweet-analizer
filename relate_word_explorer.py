import json
from collections import Counter

import networkx as nx
import nltk
import plotly.graph_objects as go
from nltk.corpus import stopwords
from pyvis.network import Network


class RelateWordExplorer:
    def __init__(self, file):
        self.dataset = json.load(file)
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words("italian"))
        self.stop_words.add("l'")
        self.stop_words.add("l’")

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

    def search_tweets(self, word):
        word = word.lower()
        return [
            tweet["id"] for tweet in self.dataset if word in tweet["tokenized_text"]
        ]

    def search_correlate_tokens(self, word):
        word = word.lower()
        correlate_ids = []
        correlate_tokens = []
        for tweet in self.dataset:
            if word in tweet["tokenized_text"]:
                correlate_ids.append(tweet["id"])
                for token in tweet["tokenized_text"]:
                    if token not in self.stop_words:
                        correlate_tokens.append(token)
        correlate_tokens = Counter(correlate_tokens)
        correlate_tokens = self.sort_dict(correlate_tokens)
        correlate_tokens = list(correlate_tokens)[0:4]
        return correlate_ids, correlate_tokens

    def search_correlate_entities(self, tweet_json, word, tool):
        correlate = []
        for tweet in tweet_json:
            if word in tweet["tokenized_text"]:
                if tool == "spacy":
                    if tweet["entity_labels"] != []:
                        for entity in tweet["entity_labels"]:
                            correlate.append(entity[1])
                if tool == "context_annotations":
                    if tweet["context_annotations"] != []:
                        for entity in tweet["context_annotations"]:
                            correlate.append(entity[1])
                if tool == "entities_annotations":
                    if tweet["entities_annotations"] != []:
                        for entity in tweet["entities_annotations"]:
                            correlate.append(entity[1])
        corr_dict = Counter(correlate)
        return {
            k: v
            for k, v in sorted(
                corr_dict.items(), key=lambda item: item[1], reverse=True
            )
        }, set(corr_dict)

    def create_correlation_graph(self, word):
        myGraph = nx.Graph()
        myGraph.add_node(word)
        myGraph = self.fill_correlation_graph(word, myGraph)
        return myGraph

    def fill_correlation_graph(self, word, graph):
        _, correlate_tokens = self.search_correlate_tokens(word)
        for token in correlate_tokens:
            if token != word and token not in list(graph.nodes):
                graph.add_node(token)
                print(f"> add node: {token}")
                self.fill_correlation_graph(token, graph)
                graph.add_edge(word, token)
            else:
                if word != token:
                    graph.add_edge(word, token)

        return graph

    def print_graph(self, G):
        net = Network(notebook=False)
        net.from_nx(G)
        net.show("example.html")


file = open("JSON/milano_finanza_2016_sumup.json")
explorer = RelateWordExplorer(file)

G = nx.Graph()
G = explorer.create_correlation_graph("brexit")

explorer.print_graph(G)
