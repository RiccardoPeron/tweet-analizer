import json
from collections import Counter
from datetime import datetime

import networkx as nx
import nltk
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from networkx.algorithms.coloring.greedy_coloring import greedy_color
from nltk.corpus import stopwords
from pyvis.network import Network


class RelateWordExplorer:
    def __init__(self, file):
        self.dataset = json.load(file)
        self.start = datetime.fromisoformat(self.dataset[0]["created_at"]).replace(
            tzinfo=None
        )
        self.end = datetime.fromisoformat(
            self.dataset[len(self.dataset) - 1]["created_at"]
        ).replace(tzinfo=None)
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

    def search_correlate_entities(self, entity, tool, start="", end=""):
        correlate = []
        if start == "":
            start = self.start
        if end == "":
            end = self.end

        if tool == "spacy":
            for tweet in self.dataset:
                if (
                    datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                    >= start
                    and datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                    <= end
                ):
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
        # correlate = list(correlate)
        # correlate = [cor for cor in list(correlate) if correlate[cor] > 1] # sfoltisce troppo forse

        return correlate

    def create_correlation_graph(self, word, data="word", mode="breadth"):
        myGraph = nx.Graph()
        if data == "word":
            myGraph.add_node(word)
            myGraph = self.fill_correlation_graph(word, myGraph, mode)
        elif data == "entity":
            myGraph.add_node(word)
            myGraph = self.fill_correlation_graph_ent(word, myGraph, mode)
        if mode == "connect":
            myGraph = self.trim_graph(myGraph, 3)
        return myGraph

    def fill_correlation_graph(self, word, graph, mode, analized=[]):
        correlate_ids, correlate_tokens = self.search_correlate_tokens(word)
        if mode == "deepth":
            for token in correlate_tokens:
                if token != word and token not in list(graph.nodes):
                    graph.add_node(token, ids=correlate_ids)
                    # print(f"> add node: {token}")
                    self.fill_correlation_graph(token, graph, "depth")
                    graph.add_edge(word, token)
        if mode == "breadth":
            for token in correlate_tokens:
                if token != word and token not in list(graph.nodes):
                    graph.add_node(token)
                    # print(f"> add node: {token}")
                    graph.add_edge(word, token)
            for token in correlate_tokens:
                if token != word and token not in analized:
                    analized.append(token)
                    self.fill_correlation_graph_v2(token, graph, "breadth", analized)

        return graph

    def fill_correlation_graph_ent(self, entity, graph, mode, analized=[]):
        correlate_ent = self.search_correlate_entities(entity, tool="spacy")
        correlate_weights = list(correlate_ent.values())
        correlate_ent = list(correlate_ent.keys())
        if mode == "depth":
            for i, ent in enumerate(correlate_ent):
                if ent != entity and ent not in list(graph.nodes):
                    graph.add_node(ent)
                    # print(f"> add node: {ent}")
                    self.fill_correlation_graph_ent(ent, graph, "depth")
                    graph.add_edge(ent, entity, weight=correlate_weights[i])
        if mode == "breadth":
            for i, ent in enumerate(correlate_ent):
                if ent != entity and ent not in list(graph.nodes):
                    graph.add_node(ent)
                    # print(f"> add node: {ent}")
                    graph.add_edge(ent, entity, weight=correlate_weights[i])
            for ent in correlate_ent:
                if ent != entity and ent not in analized:
                    analized.append(ent)
                    self.fill_correlation_graph_ent(ent, graph, "breadth", analized)
        if mode == "connect":
            for i, ent in enumerate(correlate_ent):
                if ent != entity and ent not in list(graph.nodes):
                    graph.add_node(ent)
                    # print(f"> add node: {ent}")
                    graph.add_edge(ent, entity, weight=correlate_weights[i])
                    self.fill_correlation_graph_ent(ent, graph, "connect")
                else:
                    if ent != entity:
                        # if graph.has_edge(entity, ent):
                        #     graph[entity][ent]["weight"] += 1
                        # else:
                        graph.add_edge(ent, entity, weight=correlate_weights[i])

        return graph

    def trim_graph(self, graph, node_number=1):
        weights = nx.get_edge_attributes(graph, "weight")
        edges = 0
        nodes = []
        for edge in graph.edges:
            if weights[edge] <= node_number:
                graph.remove_edge(*edge)
                edges += 1
        for node in graph.nodes:
            if list(graph.edges(node)) == []:
                nodes.append(node)
        for node in nodes:
            graph.remove_node(node)

        print(f"removed {edges} edge and {len(nodes)} node")

        return graph

    def print_graph(self, G):
        net = Network(notebook=False)
        net.from_nx(G)
        net.show("example.html")

    def get_first_occurrence(self, entity):
        for tweet in self.dataset:
            for spacy_class in ["PER", "LOC", "ORG", "MISC"]:
                for ent in tweet["entity_labels"][spacy_class]:
                    if ent == entity:
                        return tweet["created_at"]

    def monthly_correlation(self, entity):
        fig = go.Figure()

        first_occurrence = self.get_first_occurrence(entity)
        first_occurrence = datetime.fromisoformat(first_occurrence).replace(tzinfo=None)

        n_months = (self.end.year - first_occurrence.year) * 12 + (
            self.end.month - first_occurrence.month
        )

        for month in range(n_months):
            date_start = self.start + relativedelta(months=month)
            date_end = self.start + relativedelta(months=month + 1)

            month_correlate = self.search_correlate_entities(
                entity, "spacy", date_start, date_end
            )

            val = list(month_correlate.values())
            key = list(month_correlate.keys())

            for i in range(len(month_correlate)):
                found = False

                for j in range(len(fig.data)):
                    if fig.data[j].name not in key:
                        fig.data[j].x = tuple(list(fig.data[j].x) + [date_start])
                        fig.data[j].y = tuple(list(fig.data[j].y) + [0])

                    elif fig.data[j].name == key[i]:
                        found = True
                        fig.data[j].x = tuple(list(fig.data[j].x) + [date_start])
                        fig.data[j].y = tuple(list(fig.data[j].y) + [val[i]])
                        break

                if not found:
                    fig.add_trace(
                        go.Scatter(
                            x=[date_start],
                            y=[val[i]],
                            mode="lines+markers",
                            name=key[i],
                        )
                    )
        fig.show()


def explore(filename, entity, data, mode):
    file = open(filename)
    explorer = RelateWordExplorer(file)
    G = nx.Graph()
    print("creating graph...")
    G = explorer.create_correlation_graph(entity, data, mode)
    explorer.print_graph(G)


def explore2(filename, entity):
    file = open(filename)
    explorer = RelateWordExplorer(file)
    print(len(explorer.search_tweets(entity)))
    explorer.monthly_correlation(entity)


explore2("tweet_from_2016_to_2020.json", "italia")
