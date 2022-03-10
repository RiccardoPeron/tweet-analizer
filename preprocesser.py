import json
import re
import string
from datetime import datetime
import os

import spacy
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
from spacy.symbols import LOWER, ORTH

from tqdm import tqdm


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class TweeetPreprocesser:
    def __init__(
        self,
        file,
        folder,
        lang,
        start_="0001-01-01",
        end_="9999-12-31",
    ) -> None:
        print(f"\n> Preprocessing {file}...")
        file = open(file)
        self.folder = folder
        self.dataset = json.load(file)
        self.start = datetime.fromisoformat(start_).replace(tzinfo=None)
        self.end = datetime.fromisoformat(end_).replace(tzinfo=None)
        self.lang = lang
        if lang == "it":
            self.nlp = spacy.load(
                "it_core_news_lg",
                exclude=["parser", "textcat", "custom"],
            )
        elif lang == "en":
            self.nlp = spacy.load(
                "en_core_web_lg",
                exclude=["parser", "textcat", "custom"],
            )
        self.ALIASES = json.load(open("utilities/aliases.json"))


def get_tweet_data(TP):

    texts = []
    ids = []
    date = []
    context_annotations = []
    entities_annotations = []

    for data in TP.dataset:
        for tweet in data["data"]:
            if (
                datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                >= TP.start
                and datetime.fromisoformat(tweet["created_at"]).replace(tzinfo=None)
                <= TP.end
            ):
                texts.append(tweet["text"])
                ids.append(tweet["id"])
                date.append(tweet["created_at"])

                # context_annotations.append([(annotation['domain']['name'], annotation['entity']['name']) for annotation in tweet['context_annotations']])
                classes = json.load(open("utilities/context_annotation_labels.json"))
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
                        get_context_superclass(annotation["domain"]["name"], classes)
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


def get_context_superclass(context_class, classes):
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


def normalizer(text):
    text = re.sub("[\+|-]*[0-9]*[\.|,][0-9]*%", "<percent>", text)
    text = re.sub("https[:|0-9|a-z|A-Z|\.|/]*", "<url>", text)
    text = re.sub("@[a-z|A-Z]*", "<user>", text)
    text = re.sub("#[a-z|A-Z]*", "<hashtag>", text)
    text = re.sub("[(|)|/|?|!|:]", "", text)
    return text


def extract_tag(text, normalized_text):
    if (
        "<percent>" in normalized_text
        or "<hashtag>" in normalized_text
        or "<money>" in normalized_text
    ):
        percent = []
        hashtag = []
        money = []

        text = re.sub("[(|)|/|?|!|:]", "", text)
        text = text.split()
        normalized_text = normalized_text.split()

        if len(text) == len(normalized_text):
            for i in range(len(text)):
                if normalized_text[i] == "<percent>":
                    percent.append(text[i])
                    normalized_text[i] = text[i]
                if normalized_text[i] == "<hashtag>":
                    hashtag.append(text[i])
                    normalized_text[i] = text[i]
                if normalized_text[i] == "<money>":
                    money.append(text[i])
                    normalized_text[i] = text[i]
        else:
            print(f"> diff len\n{text}\n{normalized_text}")

        normalized_text = " ".join(normalized_text)

        return percent, hashtag, money, normalized_text

    return [], [], [], normalized_text


def remove_tag(text):
    text = re.sub("<.*>", "", text)
    text = text.strip()
    return text


def set_alias(word, TP):
    try:
        return TP.ALIASES[word]
    except:
        return word


def split_sentences(text):
    return re.split("\. |, |: |; |/ |-|\n", text)


def get_tokens_arg(text, TP):
    tokenized_text = []
    nouns_vetbs = []

    # sentences = split_sentences(text)
    # for text in sentences:
    nlp_text = TP.nlp(text)
    if TP.lang == "it":
        labels_structure = {"LOC": set(), "PER": set(), "ORG": set(), "MISC": set()}
    if TP.lang == "en":
        labels_structure = {"LOC": set(), "PERSON": set(), "ORG": set(), "MISC": set()}

    iobs = [token.ent_iob_ for token in nlp_text]
    tokens = [token.lower_ for token in nlp_text]
    ents = [token.ent_type_ for token in nlp_text]
    nv = [token.pos_ for token in nlp_text]
    lemmatization = [
        token.lemma_
        for token in nlp_text
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.lemma_) > 1
    ]

    for i in range(len(iobs)):
        token = []
        ent_type = ""
        if iobs[i] == "B":
            ent_type = ents[i]
            while iobs[i] != "O":
                token.append(tokens[i])
                if i < len(iobs) - 1:
                    i = i + 1
                else:
                    break
            tk = (" ".join(token)).strip()
            if len(tk) > 1:
                try:
                    labels_structure[ent_type].add(tk)
                except:
                    labels_structure["MISC"].add(tk)
                tokenized_text.append(tk)
                if nv[i] in ["NOUN", "PROPN", "VERB"]:
                    nouns_vetbs.append(tk)
        elif (
            iobs[i] == "O"
            and tokens[i] not in [".", ",", ":", ";", "#"]
            and len(tokens[i]) > 1
        ):
            tokenized_text.append(tokens[i])
            if nv[i] in ["NOUN", "PROPN", "VERB"]:
                nouns_vetbs.append(tokens[i])

    return tokenized_text, labels_structure, nouns_vetbs, lemmatization


def generate_object(text, id, date, context_annotations, entities_annotations, TP):
    ntext = normalizer(text)
    p, h, m, nt = extract_tag(text, ntext)
    nt = remove_tag(nt)
    tokens, labels, nv, lemma = get_tokens_arg(nt, TP)
    nt.lower()

    return {
        "source": TP.folder,
        "id": id,
        "created_at": date,
        "text": text,
        "normalized_text": nt,
        "tokenized_text": tokens,
        "nouns_verbs": nv,
        "lemmatization": lemma,
        "hashtags": h,
        "money": m,
        "percent": p,
        "context_annotations": context_annotations,
        "entities_annotations": entities_annotations,
        "entity_labels": labels,
    }


def generate_tweets_datas(TP):
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
    ) = get_tweet_data(TP)
    for i in tqdm(range(len(texts))):
        tweets.append(
            generate_object(
                texts[i],
                ids[i],
                dates[i],
                context_annotaions[i],
                entities_annotations[i],
                TP,
            )
        )
    tweets.reverse()

    dataset = {
        "len": len(tweets),
        "begin": tweets[0]["created_at"],
        "end": tweets[len(tweets) - 1]["created_at"],
        "tweets": tweets,
    }

    return dataset


def generate_json(tweet_datas, filename):
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
    print("Completed")


def wrap_tweet(outname, folder):
    tweet_total = []
    sources = set()

    path = os.getcwd() + "/" + folder + "/"
    print(">", path)
    for dir in os.listdir(path):
        print("\t", dir)
        filedir = path + "/" + dir
        files = []
        lengths = []
        index = []
        file_number = 0
        for file in os.listdir(filedir):
            print("\t\t", file)
            f = json.load(open(filedir + "/" + file))
            files.append(f["tweets"])
            lengths.append(len(f["tweets"]))
            index.append(0)
            file_number += 1

        tot_tweets = sum(lengths)
        for i in tqdm(range(tot_tweets)):
            tweets = []
            for j in range(file_number):
                if index[j] < lengths[j]:
                    tweets.append(files[j][index[j]]["created_at"])
                else:
                    tweets.append("9999-12-31T09:12:16+00:00")

            idx = tweets.index(min(tweets))
            tweet_total.append(files[idx][index[idx]])
            sources.add(files[idx][index[idx]]["source"])
            index[idx] += 1

    merge = {
        "len": len(tweet_total),
        "begin": tweet_total[0]["created_at"],
        "end": tweet_total[len(tweet_total) - 1]["created_at"],
        "sources": list(sources),
        "tweets": tweet_total,
    }

    with open("{}.json".format(outname), "w") as outfile:
        json.dump(merge, outfile)


def preprocess(folder_name, lan):
    path = os.getcwd() + "/" + folder_name + "/"
    try:
        os.mkdir("JSON_" + folder_name + "/")
    except:
        pass
    for dir in os.listdir(path):
        fpath = path + dir + "/"
        for file in os.listdir(fpath):
            year = file[len(dir) + 1 : len(dir) + 5]
            try:
                os.mkdir("JSON_" + folder_name + "/" + str(year))
            except:
                pass
            TP = TweeetPreprocesser(fpath + file, dir, lan)
            tweets = generate_tweets_datas(TP)
            generate_json(
                tweets, "JSON_" + folder_name + "/" + str(year) + "/" + dir + "_sumup"
            )
    print("merging preprocessed tweets...")
    wrap_tweet("merged_tweet_" + folder_name, "JSON_" + folder_name)
    return "merged_tweet_" + folder_name + ".json"


# wrap_tweet("merged_tweet_sources_it", "JSON_sources_it")
# preprocess("sources_it", "it")
# preprocess("sources_en", "en")
