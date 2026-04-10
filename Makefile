PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install train report register api

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) scripts/train_all.py --config configs/experiments.yaml

report:
	$(PYTHON) scripts/generate_report.py --experiment-name imdb-spanish-sentiment

register:
	$(PYTHON) scripts/register_best_model.py --experiment-name imdb-spanish-sentiment --registered-model-name imdb-spanish-sentiment

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8080

