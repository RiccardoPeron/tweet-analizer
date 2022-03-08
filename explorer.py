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
        self.token_number = self.count_tokens()
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

    def count_tokens(self):
        tokens_n = 0
        for tweet in self.dataset:
            tokens_n += len(tweet["lemmatization"])
        return tokens_n

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
                        if token.lower() not in self.stop_words:
                            if token.lower() in words:
                                correlate_tokens.append(words[0])
                            else:
                                correlate_tokens.append(token.lower())
                            correlate_ids.append(tweet["id"])
        correlate_tokens = Counter(correlate_tokens)
        correlate_tokens = self.sort_dict(correlate_tokens)
        # correlate_tokens = list(correlate_tokens)[0:4]
        return correlate_ids, correlate_tokens

    def create_correlation_graph(self, word, tree=True):
        myGraph = nx.Graph()
        size = len(self.search_tweets(word))

        myGraph.add_node(word[0], size=15)
        first_occurrence = self.get_first_token_occurrence(word)
        last_occourrence = self.get_last_token_occurrence(word)
        print(first_occurrence, "--->", last_occourrence)

        first_occurrence = datetime.fromisoformat(first_occurrence).replace(tzinfo=None)
        last_occourrence = datetime.fromisoformat(last_occourrence).replace(tzinfo=None)
        myGraph = self.fill_correlation_graph(
            word, size, myGraph, tree, first_occurrence, last_occourrence
        )

        if not tree:
            myGraph = self.trim_graph(myGraph, 3)
        return myGraph

    def fill_correlation_graph(
        self,
        word,
        size,
        graph,
        tree,
        first_occurrence,
        last_occourrence,
        max_neighbors=20,
        analized=[],
    ):
        min_size = 10

        if type(word) != type([]):
            word = [word]
        correlate_ids, correlate_tokens = self.search_correlate_tokens(
            word, first_occurrence, last_occourrence
        )
        correlate_weights = list(correlate_tokens.values())
        correlate_tokens = list(correlate_tokens.keys())
        word = word[0]

        if tree:
            for i, token in enumerate(correlate_tokens[:max_neighbors]):
                if (
                    token != word
                    and token not in list(graph.nodes)
                    and correlate_weights[i] > min_size
                ):
                    graph.add_node(token, size=(correlate_weights[i] / size) * 100)
                    # print(f"> add node: {token}")
                    graph.add_edge(
                        word,
                        token,
                        weight=correlate_weights[i],
                    )
            for i, token in enumerate(correlate_tokens[:max_neighbors]):
                if (
                    token != word
                    and token not in analized
                    and correlate_weights[i] > min_size
                ):
                    analized.append(token)
                    self.fill_correlation_graph(
                        token,
                        size,
                        graph,
                        tree,
                        first_occurrence,
                        last_occourrence,
                        max_neighbors,
                        analized,
                    )
        else:
            for i, token in enumerate(correlate_tokens[:max_neighbors]):
                if token != word and correlate_weights[i] > min_size:
                    graph.add_node(token, size=(correlate_weights[i] / size) * 100)
                    # print(f"> add node: {token}")
                    graph.add_edge(
                        word,
                        token,
                        weight=correlate_weights[i],
                    )
            for i, token in enumerate(correlate_tokens[:max_neighbors]):
                if (
                    token != word
                    and token not in analized
                    and correlate_weights[i] > min_size
                ):
                    analized.append(token)
                    self.fill_correlation_graph(
                        token,
                        size,
                        graph,
                        tree,
                        first_occurrence,
                        last_occourrence,
                        max_neighbors,
                        analized,
                    )

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
        net.show_buttons(filter_=["physics"])
        net.show("example.html")

    def get_last_token_occurrence(self, tokens):
        for tweet in reversed(self.dataset):
            if self.list_intersection(tokens, tweet["tokenized_text"]) > 0:
                return tweet["created_at"]

    def get_first_token_occurrence(self, tokens):
        for tweet in self.dataset:
            if self.list_intersection(tokens, tweet["tokenized_text"]) > 0:
                return tweet["created_at"]

    def monthly_correlation(self, entity, start="'", end="", day_span=15):
        dates = []

        first_occurrence = self.get_first_token_occurrence(entity)
        last_occourrence = self.get_last_token_occurrence(entity)

        if start != "" and start > first_occurrence:
            first_occurrence = datetime.fromisoformat(start).replace(tzinfo=None)
        else:
            first_occurrence = datetime.fromisoformat(first_occurrence).replace(
                tzinfo=None
            )

        if end != "" and end < last_occourrence:
            last_occourrence = datetime.fromisoformat(end).replace(tzinfo=None)
        else:
            last_occourrence = datetime.fromisoformat(last_occourrence).replace(
                tzinfo=None
            )

        print("perood: ", first_occurrence, "--->", last_occourrence)

        n_iterations = (last_occourrence - first_occurrence).days // day_span

        plot = dict()

        for iter in tqdm(range(n_iterations)):
            date_start = first_occurrence + relativedelta(days=iter * day_span)
            date_end = first_occurrence + relativedelta(
                days=(iter * day_span) + day_span
            )
            dates.append(date_start)

            _, month_correlate = self.search_correlate_tokens(
                entity, date_start, date_end
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

    def generate_plot(self, plot_dict, dates, entity, day_span):
        tot = 0
        len_plot = len(plot_dict[list(plot_dict.keys())[0]])
        for k in list(plot_dict.keys()):
            tot += sum([val for val in plot_dict[k] if val > 0])
        print("tot: ", tot)

        fig = go.Figure()
        fig.update_layout(
            title="Trends of " + entity[0],
            xaxis_title="Time",
            yaxis_title="Percentage of occurrences",
            legend_title="Tokens",
        )
        for l, line in enumerate(list(plot_dict.keys())):
            if line == entity[0]:
                fig.add_trace(
                    go.Scatter(
                        y=[(val / tot) * 100 for val in plot_dict[line]],
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
                        y=[(val / tot) * 100 for val in plot_dict[line][first:]],
                        x=dates[first:],
                        name=line,
                        mode="lines+markers",
                    )
                )

        fig.show()


def explore(filename, lan, entity, tree=True):
    print("opening file...")
    file = open(filename)
    print("explorer set up...")
    explorer = RelateWordExplorer(file, lan)
    print("counting words...")
    print(len(explorer.dataset), explorer.start, explorer.end)
    print(len(explorer.search_tweets(entity)))
    G = nx.Graph()
    print("creating graph...")
    G = explorer.create_correlation_graph(entity, tree)
    explorer.print_graph(G)


def explore2(filename, lan, entity, start="", end="", day_span=15):
    print("opening file...")
    file = open(filename)
    print("explorer set up...")
    explorer = RelateWordExplorer(file, lan)
    print("creating plot...")
    plot, dates = explorer.monthly_correlation(
        entity, start=start, end=end, day_span=day_span
    )
    print("printing plot...")
    explorer.generate_plot(plot, dates, entity, day_span)


# ITA (covid)
## explore2(
##     "merged_tweet_it.json",
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
##     start="2020-01-01",
##     day_span=15,
## )

## explore(
##     "merged_tweet_it.json",
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

# ENG (covid)
## explore2(
##     "merged_tweet_en.json",
##     "en",
##     ["covid-19", "covid", "covid 19", "coronavirus", "corona virus"],
##     day_span=15,
## )

## explore(
##     "merged_tweet_en.json",
##     "en",
##     ["covid-19", "covid", "covid 19", "coronavirus", "corona virus"],
## )
