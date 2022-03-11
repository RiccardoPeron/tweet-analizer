# tweet-analizer

## Twitter API v2

A brief description of the new version of the twitter API can be found in this document [Twitter API v2.pdf](./Twitter_API_v2.pdf)

## Setup

Create a virtual enviroment

```bash
python3 -m venv ./twitter-analizer-venv
```

Start the virtual enviroment

```bash
source twitter-analizer-venv/bin/activate
```

Install the packages

```bash
pip install -r requirements.txt
```

Download the spaCy datasets

```bash
python -m spacy download it_core_news_lg
python -m spacy download en_core_web_lg
```

## Preprocesser

Given a source folder with a structure like the one represented in the image below it outputs a `JSON` file containing all the preprocessed tweet in cronological order

```
sources_it
├── 24finanza
│   ├── 24f_2018.json
│   ├── 24f_2019.json
│   └── 24f_2020.json
├── italia_oggi
│   └── ...
├── milano_finanza
│   └── ...
└── wsi
    └── ...
```

### Functions

#### `preprocess()`

##### Parameters

| name        | type   | description                        |
| ----------- | ------ | ---------------------------------- |
| folder_name | string | name of the folder                 |
| lan         | string | language of the tweets [`it`/`en`] |

##### Example of usage

```py
import preprocesser as TP
TP.preprocess('folder_name', 'en')
```

## Analyzer

Given a file obtained from the preprocesser it outputs all the statistics of the datatset

### Functions

#### `analyze()`

##### Parameters

| name     | type   | description      |
| -------- | ------ | ---------------- |
| filename | string | name of the file |

##### Example of usage

```py
import analyzer as TA
TA.analyze("merged_tweet_sources_it.json")
```

## Explorer

With the introduction of the correlate tokens is possible to explore the relationship between the words of intrest and it correlate words.

> Correlate: token inside a sentence that contains a word of interest

### Functions

#### `correlate_graph()`

Generates a graph starting from a list of words and its correlates and proceed to analyze the correlates for each element in the correlates.

##### Parameters

| name     | type          | description                                                    |
| -------- | ------------- | -------------------------------------------------------------- |
| filename | string        | name of the file to analyze                                    |
| lan      | string        | language of the tweets [`it`/`en`]                             |
| entity   | list\<string> | list of entities to analyze                                    |
| tree     | boolean       | if true generates a tree, if false generates a connected graph |

##### Example of usage

```py
import explorer as TE
TE.correlate_graph(
  'merged_tweet_sources_it.json',
  'it',
  ['covid', 'covid-19'],
  start="2020-02-01",
  end="2020-03-30"
)
```

#### `correlate_trend()`

Given a list of words generates a line plot of the trends of the correlates.

##### Parameters

| name           | type          | description                                          |
| -------------- | ------------- | ---------------------------------------------------- |
| filename       | string        | name of the file to analyze                          |
| lan            | string        | language of the tweets [`it`/`en`]                   |
| entity         | list\<string> | list of entities to analyze                          |
| start (opt)    | string        | date of begin of the analysis in format `yyyy-mm-dd` |
| end (opt)      | string        | date of end of the analysis in format `yyyy-mm-dd`   |
| day_span (opt) | int           | number of days between two dots of the graph         |

##### Example of usage

```py
import explorer as TE
TE.correlate_trend(
  'merged_tweet_sources_it.json',
  'it',
  ['covid', 'covid-19'],
  start="2020-02-01",
  end="2020-03-30",
  day_span=5
)
```
