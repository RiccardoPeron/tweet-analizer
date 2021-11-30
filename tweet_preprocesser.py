import json
import re
import string
from datetime import datetime

import spacy
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
from spacy.symbols import LOWER, ORTH


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class TweetPreprocessing:
    def __init__(
        self,
        file,
        nlp_=spacy.load(
            "it_core_news_lg",
            exclude=["tagger", "parser", "lemmatizer", "textcat", "custom"],
        ),
        start_="0001-01-01",
        end_="9999-12-31",
    ):
        self.dataset = json.load(file)
        self.start = datetime.fromisoformat(start_).replace(tzinfo=None)
        self.end = datetime.fromisoformat(end_).replace(tzinfo=None)
        self.nlp = nlp_
        self.text_processor = TextPreProcessor(
            normalize=[
                "url",
                "email",
                "percent",
                "money",
                "phone",
                "user",
                "time",
                "date",
                "hashtag",
            ],
            unpack_hastags=False,
            dicts=[emoticons],
        )
        self.CLEANR = re.compile(" <.*?>")

    def personal_cases(self, special_cases):
        """
        Add some special cases in the tokenizer

        Parameters
        ----------
        special_cases: list of touples
        """
        for case in special_cases:
            self.nlp.tokenizer.add_special_case(case[0], case[1])

    def get_context_superclass(self, context_class, classes):
        if context_class in classes["product"]:
            return "product"
        if context_class in classes["entertainment"]:
            return "entertainment"
        if context_class in classes["videogame"]:
            return "videogame"
        if context_class in classes["sport"]:
            return "sport"
        if context_class in classes["person"]:
            return "person"
        if context_class in classes["politics"]:
            return "politics"
        if context_class in classes["brand"]:
            return "brand"
        if context_class in classes["event"]:
            return "event"
        return "other"

    def get_tweet_datas(self):
        """
        from the list of tweets returns 4 lists containing text, ids, creation_date and context_annotaions for all the tweets
        """
        # return [tweet['text'] for data in dataset for tweet in data['data']]
        if self.start > self.end:
            raise ValueError("start time can not be greater than end time")

        texts = []
        ids = []
        date = []
        context_annotations = []
        entities_annotations = []

        for data in self.dataset:
            for tweet in data["data"]:
                if (
                    datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                    >= self.start
                    and datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                    <= self.end
                ):
                    texts.append(tweet["text"])
                    ids.append(tweet["id"])
                    date.append(tweet["created_at"])

                    # context_annotations.append([(annotation['domain']['name'], annotation['entity']['name']) for annotation in tweet['context_annotations']])
                    classes = json.load(
                        open("utilities/context_annotation_labels.json")
                    )
                    annotation_schema = {
                        "product": set(),
                        "entertainment": set(),
                        "videogame": set(),
                        "sport": set(),
                        "person": set(),
                        "politics": set(),
                        "brand": set(),
                        "event": set(),
                        "other": set(),
                    }
                    for annotation in tweet["context_annotations"]:
                        annotation_schema[
                            self.get_context_superclass(
                                annotation["domain"]["name"], classes
                            )
                        ].add(annotation["entity"]["name"])
                    context_annotations.append(annotation_schema)

                    try:
                        # entities_annotations.append([(annotation['type'], annotation['normalized_text']) for annotation in tweet['entities']['annotations']])
                        annotation_schema = {
                            "Person": set(),
                            "Place": set(),
                            "Product": set(),
                            "Organization": set(),
                            "Other": set(),
                        }
                        for annotation in tweet["entities"]["annotations"]:
                            annotation_schema[annotation["type"]].add(
                                annotation["normalized_text"]
                            )
                        entities_annotations.append(annotation_schema)
                    except:
                        entities_annotations.append(
                            {
                                "Person": [],
                                "Place": [],
                                "Product": [],
                                "Organization": [],
                                "Other": [],
                            }
                        )
        return texts, ids, date, context_annotations, entities_annotations

    def normalize_text(self, text):
        """
        using the ekphrasis text prerocessor removes all the 'url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'hashtag'
        and returns the lowecase normalized text

        Parameters
        ----------
        text: string
          text to normalize
        """
        text = self.text_processor.pre_process_doc(text)
        punct = [".", ",", ";", ":", "!", "?", "(", ")"]
        for w in text:
            if w in punct:
                text = text.replace(w, "")
        return text.lower()

    def clean_tags(self, text):
        """
        using a regex removes all the HTML tags from a given text

        Parameters
        ----------
        text: string
        """
        cleantext = re.sub(self.CLEANR, "", text)
        return cleantext

    def find_normalized(self, text, norm_text):
        """
        given the normalized text returns the list of all the normalized words
        if those are <hashtags>, <money> or <percent>

        Parameters
        ----------
        text: string
          raw text
        norm_text: string
          normalized text
        """
        hashtag = []
        money = []
        percent = []
        if (
            "<hashtag>" in norm_text
            or "<money>" in norm_text
            or "<percent>" in norm_text
        ):
            for w in text:
                text = text.replace("+", "+ ")
                text = text.replace("-", "- ")
                text = text.replace("'", "' ")
                text = text.replace("’", "’ ")
            text = text.split()
            for i, word in enumerate(norm_text.split()):
                if word == "<hashtag>":
                    hashtag.append(text[i])
                    norm_text = norm_text.replace(word, text[i])
                elif word == "<money>":
                    money.append(text[i])
                    norm_text = norm_text.replace(word, text[i])
                elif word == "<percent>":
                    percent.append(text[i])
                    norm_text = norm_text.replace(word, text[i])
        return hashtag, money, percent

    def token_to_string(self, token_list, type="l"):
        """
        given a list of spacy token type objects return a list of strings of that tokens

        Parameters
        ----------
        token_list: <spacy token object> list
        type: char
          - l : retuns the lowecase text
          - n : return the raw text
        """
        if type == "l":  # all to lowercase
            return [token.lower_ for token in token_list]
        if type == "n":  # normal tetx
            return [token.text for token in token_list]

    def get_tweet_tokens_and_argouments(self, text):
        """
        given a text returns a list of touples that contains the label of the entity and the text of that entity
        using spacy

        Parameters
        ----------
        text: string
          text to analyze
        """
        # labels = []
        # texts = []
        tokens = []
        nlp_text = self.nlp(text)

        iob_list = [token.ent_iob_ for token in nlp_text]
        ent_list = [token.ent_type_ for token in nlp_text]
        tokens_list = [token for token in nlp_text]

        labels_structure = {"LOC": set(), "PER": set(), "ORG": set(), "MISC": set()}

        for i in range(len(iob_list)):
            if iob_list[i] == "B":
                b = i
                # labels.append(ent_list[i])
                label = ent_list[i]
                i += 1
                while i != len(iob_list) and iob_list[i] == "I":
                    i += 1
                text = " ".join([token.lower_ for token in tokens_list[b:i]]).strip()
                # texts.append(text)
                tokens.append(text)
                labels_structure[label].add(text)
            elif iob_list[i] == "I":
                continue
            else:
                if not tokens_list[i].is_punct:
                    tokens.append(tokens_list[i].lower_)

        return (
            tokens,
            labels_structure,
        )  # [(labels[i], texts[i]) for i in range(len(labels))]

    def generate_object(
        self, text, id, date, context_annotations, entities_annotations
    ):
        """
        given some data generates an object that represents the tweet

        Parameters
        ----------
        text: string
          text of the tweet
        id: string
          id of the tweet
        date: string
          creation_date of the tweet
        context_annotations: <string, string> list
          context_annotations list of the tweet
        """
        normalized_text = self.normalize_text(text)
        hashtags, money, percent = self.find_normalized(text, normalized_text)
        normalized_text = self.clean_tags(normalized_text)
        tokens, labels = self.get_tweet_tokens_and_argouments(normalized_text)

        return {
            "id": id,
            "created_at": date,
            "text": text,
            "normalized_text": self.clean_tags(normalized_text),
            "tokenized_text": tokens,
            "hashtags": hashtags,
            "money": money,
            "percent": percent,
            "context_annotations": context_annotations,
            "entities_annotations": entities_annotations,
            "entity_labels": labels,
        }

    def generate_tweets_datas(self):
        """
        from 4 list if text, ids, dates and context annotation generated by the get_tweet_datas function
        generates a list that contains all the nwe tweet objects

        Parameters
        ----------
        """
        tweets = []
        (
            texts,
            ids,
            dates,
            context_annotaions,
            entities_annotations,
        ) = self.get_tweet_datas()
        for i in range(len(texts)):
            tweets.append(
                self.generate_object(
                    texts[i],
                    ids[i],
                    dates[i],
                    context_annotaions[i],
                    entities_annotations[i],
                )
            )
        # print(tweets)
        return tweets

    def generate_json(self, tweet_datas, filename):
        """
        generates a JSON file from a list

        Parameters
        ----------
        tweet_datas: list
          list of tweet generated by the generate_tweets_datas functon
        filename: string
          name of the output file
        """
        with open("{}.json".format(filename), "w") as outfile:
            json.dump(tweet_datas, outfile, cls=SetEncoder)


def preprocess(filename, outputname):
    tweet_file = open(filename)
    preprocesser = TweetPreprocessing(tweet_file)
    preprocesser.personal_cases(
        [
            ("piazza affari", [{"ORTH": "piazza affari"}]),
            #    ("wall street", [{"ORTH": "wall street"}]),
        ]
    )
    data = preprocesser.generate_tweets_datas()
    preprocesser.generate_json(data, outputname)
    return data


preprocess(
    "milano_finanza/milano_finanza_2016-01-01T00_00_00Z_2016-12-31T23_59_59Z.json",
    "JSON/milano_finanza_2016_sumup",
)
preprocess(
    "milano_finanza/milano_finanza_2017-01-01T00_00_00Z_2017-12-31T23_59_59Z.json",
    "JSON/milano_finanza_2017_sumup",
)
preprocess(
    "milano_finanza/milano_finanza_2018-01-01T00_00_00Z_2018-12-31T23_59_59Z.json",
    "JSON/milano_finanza_2018_sumup",
)
preprocess(
    "milano_finanza/milano_finanza_2019-01-01T00_00_00Z_2019-12-31T23_59_59Z.json",
    "JSON/milano_finanza_2019_sumup",
)
preprocess(
    "milano_finanza/milano_finanza_2020-01-01T00_00_00Z_2020-12-31T23_59_59Z.json",
    "JSON/milano_finanza_2020_sumup",
)
