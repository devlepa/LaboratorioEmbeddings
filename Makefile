PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install train report register select api

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

# Muestra el mejor modelo disponible localmente y su ruta.
select:
	$(PYTHON) scripts/select_best_model.py --verbose

# Lanza la API usando el mejor modelo encontrado en los artefactos locales.
# MODEL_URI puede sobreescribirse:  make api MODEL_URI=<ruta_o_uri>
MODEL_URI ?= $(shell $(PYTHON) scripts/select_best_model.py)
api:
	MODEL_URI="$(MODEL_URI)" PYTHONPATH=src $(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8080

