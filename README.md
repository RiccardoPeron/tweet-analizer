# tweet-analizer

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
python -m spacy download en_core_web_trf
```
