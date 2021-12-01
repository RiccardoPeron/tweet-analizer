import json
from collections import Counter

import networkx as nx
from networkx.algorithms.coloring.greedy_coloring import greedy_color
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
                for token in tweet["tokenized_text"]:
                    if token not in self.stop_words:
                        correlate_tokens.append(token)
                        correlate_ids.append(tweet["id"])
        correlate_tokens = Counter(correlate_tokens)
        correlate_tokens = self.sort_dict(correlate_tokens)
        correlate_tokens = list(correlate_tokens)[0:4]
        return correlate_ids, correlate_tokens

    def search_correlate_entities(self, entity, tool):
        correlate = []

        if tool == "spacy":
            for tweet in self.dataset:
                entities = []
                isent = False
                for ent in tweet["entity_labels"]["PER"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["entity_labels"]["LOC"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["entity_labels"]["ORG"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["entity_labels"]["MISC"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                if isent:
                    for ent in entities:
                        correlate.append(ent)
        if tool == "context_annotations":
            for tweet in self.dataset:
                entities = []
                isent = False
                for ent in tweet["context_annotations"]["product"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["context_annotations"]["entertainment"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["context_annotations"]["videogame"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["context_annotations"]["sport"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["context_annotations"]["person"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["context_annotations"]["politics"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["context_annotations"]["brand"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["context_annotations"]["event"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["context_annotations"]["other"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                if isent:
                    for ent in entities:
                        correlate.append(ent)
        if tool == "entities_annotations":
            for tweet in self.dataset:
                entities = []
                isent = False
                for ent in tweet["entities_annotations"]["Person"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["entities_annotations"]["Place"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["entities_annotations"]["Product"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["entities_annotations"]["Organization"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)
                for ent in tweet["entities_annotations"]["Other"]:
                    if entity == ent:
                        isent = True
                    entities.append(ent)

        correlate = Counter(correlate)
        correlate = self.sort_dict(correlate)
        correlate = list(correlate)

        return correlate

    def create_correlation_graph(self, word, data="word"):
        myGraph = nx.Graph()
        if data == "word":
            myGraph.add_node(word)
            myGraph = self.fill_correlation_graph_v2(word, myGraph)
        elif data == "entity":
            myGraph.add_node(word)
            myGraph = self.fill_correlation_graph_ent(word, myGraph)
        return myGraph

    def fill_correlation_graph(self, word, graph):
        correlate_ids, correlate_tokens = self.search_correlate_tokens(word)
        for token in correlate_tokens:
            if token != word and token not in list(graph.nodes):
                graph.add_node(token, ids=correlate_ids)
                print(f"> add node: {token}")
                self.fill_correlation_graph(token, graph)
                graph.add_edge(word, token)
            else:
                if word != token:
                    graph.add_edge(word, token)

        return graph

    def fill_correlation_graph_v2(self, word, graph, analized=[]):
        _, correlate_tokens = self.search_correlate_tokens(word)
        for token in correlate_tokens:
            if token != word and token not in list(graph.nodes):
                graph.add_node(token)
                print(f"> add node: {token}")
                graph.add_edge(word, token)
        for token in correlate_tokens:
            if token != word and token not in analized:
                analized.append(token)
                self.fill_correlation_graph_v2(token, graph, analized)

        return graph

    def fill_correlation_graph_ent(self, entity, graph, analized=[]):
        correlate_ent = self.search_correlate_entities(
            entity, tool="entities_annotations"
        )
        for ent in correlate_ent:
            if ent != entity and ent not in list(graph.nodes):
                graph.add_node(ent)
                print(f"> add node: {ent}")
                # self.fill_correlation_graph_ent(ent, graph)
                graph.add_edge(ent, entity)
            # else:
            #     if ent != entity:
            #         graph.add_edge(entity, ent)
        for ent in correlate_ent:
            if ent != entity and ent not in analized:
                analized.append(ent)
                self.fill_correlation_graph_ent(ent, graph)

        return graph

    def print_graph(self, G):
        net = Network(notebook=False)
        net.from_nx(G)
        net.show("example.html")


file = open("JSON/milano_finanza_2016_sumup.json")
explorer = RelateWordExplorer(file)

G = nx.Graph()
G = explorer.create_correlation_graph("brexit", "entity")

explorer.print_graph(G)
