#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m ipykernel install --user --name imdb-spanish-sentiment --display-name "Python (IMDB Spanish Sentiment)"
python -m spacy download es_core_news_md
python -m spacy download es_core_news_lg

echo "Entorno listo. Actívalo con: source .venv/bin/activate"

