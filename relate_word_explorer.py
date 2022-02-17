import json
from collections import Counter
from datetime import datetime

import networkx as nx
import nltk
from nltk.corpus import stopwords
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from networkx.algorithms.coloring.greedy_coloring import greedy_color
from nltk.corpus import stopwords
from pyvis.network import Network
from tqdm import tqdm


class RelateWordExplorer:
    def __init__(self, file, lan):
        self.dataset = json.load(file)
        self.start = datetime.fromisoformat(self.dataset[0]["created_at"]).replace(
            tzinfo=None
        )
        self.end = datetime.fromisoformat(
            self.dataset[len(self.dataset) - 1]["created_at"]
        ).replace(tzinfo=None)
        nltk.download("stopwords")
        if lan == "it":
            self.stop_words = set(stopwords.words("italian"))
            self.stop_words.add("l'")
            self.stop_words.add("l’")
        if lan == "en":
            self.stop_words = set(stopwords.words("english"))

    def list_intersection(self, l1, l2):
        return len(set(l1) & set(l2))

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

    def search_tweets(self, words):
        words = [word.lower() for word in words]
        return [
            tweet["id"]
            for tweet in self.dataset
            if len(set(words) & set(tweet["tokenized_text"])) > 0
        ]

    def search_correlate_tokens(self, words, start="", end=""):
        if start == "":
            start = self.start
        if end == "":
            end = self.end

        words = [word.lower() for word in words]
        correlate_ids = []
        correlate_tokens = []
        for tweet in self.dataset:
            if (
                datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                >= start
                and datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                <= end
            ):
                if self.list_intersection(words, tweet["tokenized_text"]) > 0:
                    for token in tweet["lemmatization"]:
                        if token not in self.stop_words:
                            if token in words:
                                correlate_tokens.append(words[0])
                            else:
                                correlate_tokens.append(token)
                            correlate_ids.append(tweet["id"])
        correlate_tokens = Counter(correlate_tokens)
        correlate_tokens = self.sort_dict(correlate_tokens)
        # correlate_tokens = list(correlate_tokens)[0:4]
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
                    if entity in tweet["tokenized_text"]:
                        for ent in tweet["entity_labels"]["PER"]:
                            correlate.append(ent)
                        for ent in tweet["entity_labels"]["LOC"]:
                            correlate.append(ent)
                        for ent in tweet["entity_labels"]["ORG"]:
                            correlate.append(ent)
                        for ent in tweet["entity_labels"]["MISC"]:
                            correlate.append(ent)

                    ## entities = []
                    ## isent = False
                    ## for ent in tweet["entity_labels"]["PER"]:
                    ##     if entity == ent:
                    ##         isent = True
                    ##     entities.append(ent)
                    ## for ent in tweet["entity_labels"]["LOC"]:
                    ##     if entity == ent:
                    ##         isent = True
                    ##     entities.append(ent)
                    ## for ent in tweet["entity_labels"]["ORG"]:
                    ##     if entity == ent:
                    ##         isent = True
                    ##     entities.append(ent)
                    ## for ent in tweet["entity_labels"]["MISC"]:
                    ##     if entity == ent:
                    ##         isent = True
                    ##     entities.append(ent)
                    ## if isent:
                    ##     for ent in entities:
                    ##         correlate.append(ent)
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
        correlate_tokens = list(correlate_tokens.keys())
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
                    print(f"> add node: {token}")
                    graph.add_edge(word, token)
            for token in correlate_tokens:
                if token != word and token not in analized:
                    analized.append(token)
                    self.fill_correlation_graph(token, graph, "breadth", analized)

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

    def get_last_token_occurrence(self, tokens):
        for tweet in reversed(self.dataset):
            if self.list_intersection(tokens, tweet["tokenized_text"]) > 0:
                return tweet["created_at"]

    def get_first_token_occurrence(self, tokens):
        for tweet in self.dataset:
            if self.list_intersection(tokens, tweet["tokenized_text"]) > 0:
                return tweet["created_at"]

    def monthly_correlation(self, entity, day_span=15, mode="token"):
        dates = []

        first_occurrence = self.get_first_token_occurrence(entity)
        # last_occourrence = self.get_last_token_occurrence(entity)

        first_occurrence = datetime.fromisoformat(first_occurrence).replace(tzinfo=None)
        # last_occourrence = datetime.fromisoformat(last_occourrence).replace(tzinfo=None)

        n_iterations = (self.end - first_occurrence).days // day_span

        plot = dict()

        for iter in tqdm(range(n_iterations)):
            date_start = first_occurrence + relativedelta(days=iter * day_span)
            date_end = first_occurrence + relativedelta(
                days=(iter * day_span) + day_span
            )
            dates.append(date_start)

            if mode == "token":
                _, month_correlate = self.search_correlate_tokens(
                    entity, date_start, date_end
                )
            else:
                month_correlate = self.search_correlate_entities(
                    entity, "spacy", date_start, date_end
                )

            vals = list(month_correlate.values())[:5]
            keys = list(month_correlate.keys())[:5]

            for i, key in enumerate(keys):
                if key not in list(plot.keys()):
                    plot[key] = [-1] * n_iterations
                if key in list(plot.keys()):
                    plot[key][iter] = vals[i]

            for line in plot.keys():
                if plot[line][iter] == -1:
                    plot[line][iter] = 0

        return plot, dates

    def generate_plot(self, plot_dict, dates, entity):
        fig = go.Figure()
        for line in list(plot_dict.keys()):
            if line == entity[0]:
                fig.add_trace(
                    go.Scatter(
                        y=plot_dict[line],
                        x=dates,
                        name=line,
                        mode="lines+markers",
                    )
                )
            elif plot_dict[line].count(0) < 10:
                first = 0
                for i, el in enumerate(plot_dict[line]):
                    if el > 0:
                        first = i
                        break
                fig.add_trace(
                    go.Scatter(
                        y=plot_dict[line][first:],
                        x=dates[first:],
                        name=line,
                        mode="lines+markers",
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


def explore2(filename, lan, entity):
    print("opening file...")
    file = open(filename)
    print("explorer set up...")
    explorer = RelateWordExplorer(file, lan)
    print("counting words...")
    print(len(explorer.dataset), explorer.start, explorer.end)
    print(len(explorer.search_tweets(entity)))
    print("creating plot...")
    plot, dates = explorer.monthly_correlation(entity, mode="token")
    print("printing plot...")
    explorer.generate_plot(plot, dates, entity)


# explore("tweet_from_2016_to_2020.json", "covid-19", "word", "breadth")
## explore2(
##     "tweet_from_2016_to_2020.json",
##     "it",
##     [
##         "covid-19",
##         "covid",
##         "covid 19",
##         "coronavirus",
##         "corona virus",
##         "pandemia",
##         "epidemia",
##     ],
## )

## explore2(
##     "tweet_from_2016_to_2020_en.json",
##     "en",
##     [
##         "covid-19",
##         "covid",
##         "covid 19",
##         "coronavirus",
##         "corona virus",
##         "pandemy",
##         "epidemy",
##     ],
## )
