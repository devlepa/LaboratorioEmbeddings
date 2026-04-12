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

## Variables de entorno

Copiá el archivo de plantilla y completá los valores reales antes de cualquier otro paso:

```bash
cp .env.example .env
```

Editá `.env` con tu DNS público de Cloud9, tus credenciales de Kaggle y el URI del modelo. El DNS lo encontrás en el botón **Preview** de Cloud9.

> `.env` está en `.gitignore` y nunca se sube al repositorio.

## Entorno virtual

```bash
bash scripts/create_venv.sh
source .venv/bin/activate
```

Si preferís hacerlo manualmente:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_md
python -m spacy download es_core_news_lg
python -m ipykernel install --user --name imdb-spanish-sentiment --display-name "Python (IMDB Spanish Sentiment)"
```

## Flujo completo de ejecución

Cargá las variables de entorno al inicio de cada terminal nueva:

```bash
source .env
```

Los pasos deben ejecutarse en este orden.

### 1. Levantar el servidor de MLflow (terminal dedicada)

```bash
source .env
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

Dejá esta terminal corriendo. La UI queda disponible en `http://<CLOUD9_PUBLIC_DNS>:5000`.

### 2. Ejecutar los experimentos (nueva terminal)

```bash
source .env && source .venv/bin/activate
python scripts/train_all.py --config configs/experiments.yaml
```

Para correr sólo una parte:

```bash
python scripts/train_all.py --config configs/experiments.yaml \
  --only baseline_tfidf_logreg dense_trainable_embeddings
```

### 3. Generar el reporte comparativo

```bash
python scripts/generate_report.py --experiment-name imdb-spanish-sentiment
```

Esto genera:

- `artifacts/reports/comparison.csv`
- `artifacts/reports/comparative_report.md`
- `artifacts/reports/test_f1_comparison.png`

### 4. Registrar el mejor modelo

```bash
python scripts/register_best_model.py \
  --experiment-name imdb-spanish-sentiment \
  --registered-model-name imdb-spanish-sentiment
```

El alias `champion` queda asignado automáticamente. El URI del modelo es siempre:

```
models:/imdb-spanish-sentiment@champion
```

### 5. Levantar la API (nueva terminal)

```bash
source .env && source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Si querés apuntar a una versión numérica específica en lugar del alias, editá `MODEL_URI` en `.env`:

```
MODEL_URI=models:/imdb-spanish-sentiment/3
```

### 6. Verificar que la API responde

```bash
curl -X POST "http://<CLOUD9_PUBLIC_DNS>:8000/predict" \
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
