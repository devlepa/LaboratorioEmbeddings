# LaboratorioEmbeddings

Proyecto base para el laboratorio de análisis de sentimientos sobre el dataset **IMDB Movie Reviews en español**. El repositorio quedó preparado para cubrir:

- baseline con **TF-IDF + Regresión Logística**
- red neuronal densa con **embeddings entrenables**
- tres fuentes de **embeddings preentrenados** con dos variantes cada una: congelados y ajustados
- registro completo de corridas en **MLflow**
- análisis comparativo automatizable
- despliegue de la mejor versión mediante **FastAPI**, cargando el modelo desde MLflow

## Estructura

- `configs/experiments.yaml`: configuración central del proyecto
- `src/imdb_sentiment/`: paquete con carga de datos, modelos, embeddings, evaluación y tracking
- `scripts/train_all.py`: ejecuta todos los experimentos
- `scripts/generate_report.py`: genera el análisis comparativo a partir de MLflow
- `scripts/register_best_model.py`: registra el mejor modelo en MLflow
- `api/main.py`: API para despliegue en Cloud9
- `notebooks/`: notebooks explicativos del flujo completo

## Entorno virtual

1. Crear entorno:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_md
python -m spacy download es_core_news_lg
```

3. Registrar kernel para Jupyter:

```bash
python -m ipykernel install --user --name imdb-spanish-sentiment --display-name "Python (IMDB Spanish Sentiment)"
```

También puedes usar:

```bash
bash scripts/create_venv.sh
```

## Ejecución de experimentos

Define primero el tracking de MLflow si vas a usar una máquina dedicada:

```bash
export MLFLOW_TRACKING_URI="http://TU_SERVIDOR_MLFLOW:5000"
```

Luego ejecuta:

```bash
python scripts/train_all.py --config configs/experiments.yaml
```

Para correr sólo una parte:

```bash
python scripts/train_all.py --config configs/experiments.yaml --only baseline_tfidf_logreg dense_trainable_embeddings
```

## Reporte comparativo

```bash
python scripts/generate_report.py --experiment-name imdb-spanish-sentiment
```

Esto genera:

- `artifacts/reports/comparison.csv`
- `artifacts/reports/comparative_report.md`
- `artifacts/reports/test_f1_comparison.png`

## Registro del mejor modelo

```bash
python scripts/register_best_model.py \
  --experiment-name imdb-spanish-sentiment \
  --registered-model-name imdb-spanish-sentiment
```

La API usa por defecto:

```bash
models:/imdb-spanish-sentiment@champion
```

## API para Cloud9

Variables recomendadas:

```bash
export MLFLOW_TRACKING_URI="http://TU_SERVIDOR_MLFLOW:5000"
export MODEL_URI="models:/imdb-spanish-sentiment@champion"
```

Levantar servicio:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Ejemplo de consumo:

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["La película fue excelente", "Una historia aburrida y lenta"]}'
```

## Justificación de la arquitectura neuronal

La arquitectura definida en `configs/experiments.yaml` mantiene constante la parte densa en todos los experimentos neuronales:

- `sequence_length = 300`: suficiente para captar gran parte del contexto de una reseña sin disparar innecesariamente el costo computacional.
- `embedding_dim = 128` para el modelo entrenable: compromiso razonable entre capacidad y riesgo de sobreajuste.
- `hidden_units = [128, 64]`: dos capas ocultas permiten capturar interacciones no lineales tras el pooling sin volver el modelo excesivamente profundo.
- `dropout = 0.30`: regularización simple y efectiva para una tarea binaria con 50k reseñas.

En los modelos con embeddings preentrenados sólo cambia la capa de embeddings, tal como pide la consigna.

## Nota sobre el número de experimentos

La consigna menciona **7 experimentos**, pero al mismo tiempo pide:

- 1 baseline
- 1 red con embeddings entrenables
- 3 embeddings preentrenados × 2 variantes (congelado y ajustado) = 6

Eso suma **8 corridas**. El pipeline quedó preparado para ejecutar las 8, porque así se cumplen todos los requisitos textuales. Si el docente exige estrictamente 7, basta con desactivar una corrida en `configs/experiments.yaml`.
