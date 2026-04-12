#!/usr/bin/env bash
set -euo pipefail

rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 -m ipykernel install --user --name imdb-spanish-sentiment --display-name "python3 (IMDB Spanish Sentiment)"
python3 -m spacy download es_core_news_md
python3 -m spacy download es_core_news_lg

echo "Entorno listo. Actívalo con: source .venv/bin/activate"

