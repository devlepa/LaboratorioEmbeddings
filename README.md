# IMDB Spanish Sentiment — Laboratorio de Embeddings

Análisis de sentimientos en español sobre reseñas de cine (IMDB). Compara 8 experimentos —un baseline clásico contra redes neuronales con distintos embeddings— registra todo en MLflow y expone el modelo ganador via FastAPI.

**Modelo en producción:** `baseline_tfidf_logreg` (TF-IDF + Regresión Logística)  
**F1 macro (test):** 0.8996 &nbsp;|&nbsp; **ROC-AUC:** 0.9626

Servicios desplegados en EC2:
- MLflow UI: `http://ec2-13-223-188-95.compute-1.amazonaws.com:5000`
- API REST:  `http://ec2-13-223-188-95.compute-1.amazonaws.com:8000`

---

## Estructura de archivos

```
.
├── api/
│   └── main.py                  # FastAPI: /health  /model-info  /predict
│
├── configs/
│   └── experiments.yaml         # Hiperparámetros y definición de los 8 experimentos
│
├── models/
│   └── champion/                # Modelo ganador listo para servir (comprometido en git)
│       ├── MLmodel              # Metadatos MLflow: run_id, firma entrada/salida
│       ├── python_model.pkl     # Wrapper SentimentPyfuncModel serializado
│       └── artifacts/
│           ├── baseline_pipeline.joblib   # Pipeline sklearn entrenado
│           ├── metadata.json              # backend, threshold, model_name
│           └── analysis/
│               ├── test_classification_report.json
│               ├── validation_classification_report.json
│               ├── dataset_metadata.json
│               ├── experiment_config.json
│               └── metrics_summary.json   # roc_auc y métricas agregadas
│
├── scripts/
│   ├── train_all.py             # Entrena experimentos y los sube a MLflow
│   ├── select_best_model.py     # Elige el mejor modelo desde mlruns/ local
│   ├── register_best_model.py   # Registra el mejor en el Model Registry de MLflow
│   ├── generate_report.py       # CSV + gráfico + markdown comparativo
│   └── migrate_runs.py          # Utilidad: migra runs entre servidores MLflow
│
├── src/imdb_sentiment/
│   ├── config.py                # Dataclasses de configuración + load_config()
│   ├── data.py                  # Carga Kaggle y genera splits train/val/test
│   ├── preprocessing.py         # normalize_text() y TextVectorization de Keras
│   ├── modeling.py              # build_baseline_pipeline() y build_neural_model()
│   ├── embeddings.py            # build_embedding_matrix() para spaCy y gensim
│   └── tracking.py              # SentimentPyfuncModel + log_deployable_model()
│
├── Makefile
└── requirements.txt
```

> **`mlruns/` no está en git** (está en `.gitignore`). Las métricas y artefactos de análisis de cada run se guardan en el servidor MLflow remoto. El directorio `models/champion/artifacts/analysis/` sí está en git y contiene todos los JSONs que necesita la API.

---

## Cómo funciona: flujo completo

### 1. Configuración (`configs/experiments.yaml`)

Un único YAML controla todo el proyecto:

```yaml
dataset:        # Handle de Kaggle y nombres de columnas candidatos
splits:         # 80% train / 10% val / 10% test, semilla 42
baseline:       # max_features=50000, ngram_range=[1,2], C=4.0
neural:         # embedding_dim=128, hidden_units=[128,64], dropout=0.30, epochs=12
tracking:       # experiment_name, registered_model_name
experiments:    # lista de los 8 experimentos con sus embeddings
```

`load_config(path)` en `config.py` convierte el YAML en dataclasses tipadas (`ProjectConfig`, `ExperimentConfig`, etc.).

---

### 2. Datos (`src/imdb_sentiment/data.py`)

```
Kaggle: luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish
   │
   ├─ _find_dataset_file()   →  busca en cache ~/.cache/kagglehub/
   │                             si no existe, descarga con kagglehub
   │
   ├─ pd.read_csv()
   ├─ normalize_text()        →  minúsculas, sin HTML ni puntuación
   ├─ _normalize_labels()     →  "positive"/"negative" → 1/0
   │
   └─ train_test_split() ×2  →  80% train / 10% val / 10% test  (estratificado)
                                  → DataSplit(x_train, y_train, x_val, y_val, x_test, y_test)
```

Dataset: 50.000 reseñas perfectamente balanceadas (25k positivas / 25k negativas).

---

### 3. Preprocesamiento (`src/imdb_sentiment/preprocessing.py`)

`normalize_text(text)` — versión Python (usada en el pipeline sklearn y en la limpieza inicial):

```
html.unescape → elimina tags HTML → elimina puntuación → colapsa espacios → lower()
```

`keras_standardization(tensor)` — versión TensorFlow (idéntica lógica, opera sobre tensores). Se usa como función `standardize=` dentro de la capa `TextVectorization` de Keras.

---

### 4. Entrenamiento (`scripts/train_all.py`)

Por cada experimento definido en el YAML, abre un run de MLflow y toma una de dos rutas:

#### Ruta baseline (TF-IDF + LogReg)

```python
pipeline = build_baseline_pipeline(config.baseline)
# Pipeline([TfidfVectorizer(preprocessor=normalize_text, ngrams=(1,2)), LogisticRegression(C=4)])

pipeline.fit(x_train, y_train)

probs = pipeline.predict_proba(x_test)[:, 1]
metrics = binary_metrics(y_test, probs, threshold=0.5)
# → test_accuracy, test_f1, test_precision, test_recall, test_roc_auc

mlflow.log_metrics(metrics)
mlflow.log_artifacts(analysis_dir)      # JSONs + confusion matrix + ROC curve
log_deployable_model(pipeline, backend="sklearn")
```

#### Ruta neural (Keras)

```python
# 1. Vectorización de texto
vectorizer = adapt_vectorizer(x_train, config.neural)
# TextVectorization con keras_standardization adaptado al vocabulario de entrenamiento

# 2. Embeddings (solo si el experimento usa preentrenados)
matrix = build_embedding_matrix(vocabulary, experiment.embedding)
# spaCy: carga es_core_news_md/lg → vector por token (~84% cobertura)
# gensim/ConceptNet: busca /c/es/<token> (~33% cobertura)

# 3. Arquitectura
model = build_neural_model(config.neural, vectorizer, matrix)
# Input(text) → TextVectorization → Embedding → GlobalAvgPooling → Dense(128) → Dense(64) → sigmoid

# 4. Entrenamiento
history = fit_neural_model(model, x_train, y_train, x_val, y_val, config.neural)
# EarlyStopping(monitor="val_loss", patience=3), restaura mejores pesos

mlflow.log_metrics(metrics)
mlflow.log_artifacts(analysis_dir)
log_deployable_model(model, backend="keras")
```

**Artefactos guardados por run en MLflow:**

| Archivo | Descripción |
|---------|-------------|
| `analysis/test_classification_report.json` | Reporte completo por clase (precisión, recall, F1) |
| `analysis/validation_classification_report.json` | Ídem en validation set |
| `analysis/dataset_metadata.json` | Fuente, filas, distribución de clases |
| `analysis/experiment_config.json` | Nombre y tipo del experimento |
| `analysis/test_confusion_matrix.png` | Matriz de confusión |
| `analysis/test_roc_curve.png` | Curva ROC |
| `analysis/test_predictions.csv` | Probabilidades y predicciones del test set |
| `model/` | Modelo serializado + metadata.json + código fuente |

---

### 5. Módulo de tracking (`src/imdb_sentiment/tracking.py`)

Resuelve el problema de que sklearn y Keras tienen APIs de inferencia distintas pero la API REST necesita una interfaz uniforme.

**`SentimentPyfuncModel(mlflow.pyfunc.PythonModel)`**:
- `load_context()`: carga el `.joblib` (sklearn) o el `.keras` (Keras) según `backend` en `metadata.json`
- `predict(df)`: acepta `DataFrame{"text": [...]}`, devuelve `[text, probability, predicted_class, predicted_label]`

**`log_deployable_model(model, backend, artifact_path, metadata)`**:
1. Serializa el modelo (`.joblib` o `.keras`)
2. Infiere la firma de entrada/salida con un ejemplo
3. Sube a MLflow: modelo + metadata.json + código de `src/` (para que el wrapper sea importable al servir)

---

### 6. Embeddings (`src/imdb_sentiment/embeddings.py`)

`build_embedding_matrix(vocabulary, config)` → numpy array `(vocab_size, embedding_dim)`:

- **spaCy** (`es_core_news_md` / `lg`): busca el vector de cada token en el vocabulario spaCy. Cobertura ~84%.
- **ConceptNet Numberbatch** (gensim): busca la clave `/c/es/<token>` en el grafo. Cobertura ~33% porque el vocabulario de reseñas tiene muchos términos fuera del grafo de conocimiento.

Los modelos de embeddings se cachean con `@lru_cache` para no recargarlos entre experimentos.

---

### 7. Resultados de los 8 experimentos

| Modelo | F1 (test) | ROC-AUC | Cobertura |
|--------|-----------|---------|-----------|
| **baseline_tfidf_logreg** | **0.8996** | **0.9626** | 100% |
| dense_trainable_embeddings | 0.8806 | — | — |
| conceptnet_finetuned | 0.8804 | 0.9493 | 32.62% |
| spacy_lg_finetuned | 0.8768 | — | 84.72% |
| spacy_md_finetuned | 0.8695 | — | 84.72% |
| conceptnet_frozen | 0.8014 | 0.8873 | 32.62% |
| spacy_md_frozen | 0.7920 | — | 84.72% |
| spacy_lg_frozen | 0.7905 | — | 84.72% |

**Por qué ganó el baseline:**
- TF-IDF cubre el 100% del vocabulario; los embeddings preentrenados dejan sin representación entre el 15% y el 68% de los tokens.
- Las reseñas usan vocabulario polarizado explícito (*excelente*, *horrible*, *magnífico*) que los bigramas capturan directamente.
- Los embeddings congelados pierden hasta 10 puntos de F1 por los tokens sin vector.
- Afinar los embeddings mejora, pero no supera al baseline: la complejidad adicional no aporta valor en este corpus.
- El baseline es determinista, más rápido en inferencia y no requiere GPU.

---

### 8. API REST (`api/main.py`)

Al arrancar, el `lifespan` carga el modelo una sola vez:

```python
_model = mlflow.pyfunc.load_model(MODEL_URI)   # "models/champion" por defecto
_model_info = _load_model_info()               # Lee JSONs de models/champion/artifacts/analysis/
```

**Endpoints:**

#### `GET /health`
```json
{"status": "ok", "model_uri": "models/champion"}
```

#### `GET /model-info`
Devuelve metadatos completos del modelo en producción:
```json
{
  "model_name": "baseline_tfidf_logreg",
  "backend": "sklearn",
  "threshold": 0.5,
  "dataset": {"source": "...", "total_rows": 50000, "splits": {...}},
  "test_metrics": {"accuracy": 0.8996, "f1": 0.8996, "roc_auc": 0.9626, ...},
  "test_metrics_per_class": {
    "negative": {"precision": 0.9064, "recall": 0.8912, "f1": 0.8987, "support": 2500},
    "positive": {"precision": 0.8930, "recall": 0.9080, "f1": 0.9004, "support": 2500}
  },
  "validation_metrics": {...},
  "experiment_comparison": [...],
  "selection_criteria": {"metric": "test_f1 (macro avg)", "rationale": [...]}
}
```

Las métricas se leen desde `models/champion/artifacts/analysis/` (en git). Si no existen, hace fallback al servidor MLflow via `MlflowClient`.

#### `POST /predict`
```bash
curl -X POST http://.../predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Una película magnífica", "Terrible pérdida de tiempo"]}'
```
```json
{"predictions": [
  {"text": "Una película magnífica", "probability": 0.923, "predicted_class": 1, "predicted_label": "positive"},
  {"text": "Terrible pérdida de tiempo", "probability": 0.071, "predicted_class": 0, "predicted_label": "negative"}
]}
```

---

## Setup y comandos

### Setup inicial
```bash
make venv && make install
```

### Variables de entorno
```bash
export MLFLOW_TRACKING_URI=http://ec2-13-223-188-95.compute-1.amazonaws.com:5000
export MODEL_URI=models/champion          # o: models:/imdb-spanish-sentiment@champion
```

### Comandos frecuentes

```bash
# Entrenar todos los experimentos
make train

# Entrenar solo uno
.venv/bin/python scripts/train_all.py --only baseline_tfidf_logreg

# Ver el mejor modelo disponible localmente (desde mlruns/)
make select

# Actualizar models/champion/ con el mejor modelo local
make update-champion

# Registrar el mejor modelo en MLflow Model Registry
make register

# Generar reporte comparativo (CSV + gráfico + markdown)
make report

# Levantar la API localmente (puerto 8080)
make api
```

---

## Cómo actualizar el modelo en producción

```
1. Entrenar nuevos experimentos
   → scripts/train_all.py sube todo a MLflow (EC2:5000)

2. Seleccionar y copiar el mejor
   → make update-champion
      (copia el modelo ganador a models/champion/)

3. Copiar los JSONs de análisis
   → cp mlruns/1/<run_id>/artifacts/analysis/*.json models/champion/artifacts/analysis/
   → editar metrics_summary.json con los nuevos roc_auc

4. Subir a git
   → git add models/champion/ && git commit && git push

5. En EC2/Cloud9
   → git pull
   → reiniciar uvicorn para que cargue el nuevo modelo
```

---

## Decisiones de diseño

**¿Por qué `models/champion/` está en git?**  
El modelo pesa ~2.3 MB. Tenerlo en git elimina la dependencia del servidor MLflow para levantar la API y hace el despliegue más simple y robusto.

**¿Por qué los JSONs de análisis están dentro de `models/champion/`?**  
`mlruns/` está en `.gitignore`. Si los JSONs solo existieran en `mlruns/`, el endpoint `/model-info` devolvería ceros en EC2 después de un `git pull`. Copiarlos dentro del directorio del modelo (que sí está en git) garantiza que la API siempre tenga las métricas completas.

**¿Por qué `SentimentPyfuncModel` en vez de cargar el modelo directamente?**  
MLflow requiere un `PythonModel` para empaquetar el modelo con su lógica de inferencia. Esto permite una interfaz unificada independientemente de si el backend es sklearn o Keras, y facilita servir el modelo desde cualquier herramienta compatible con MLflow pyfunc.

**¿Por qué 8 experimentos en lugar de 7?**  
La consigna pide: 1 baseline + 1 red con embeddings entrenables + 3 fuentes de embeddings preentrenados × 2 variantes (congelado/ajustado) = 1+1+6 = 8. Si se requieren exactamente 7, basta con comentar uno en `configs/experiments.yaml`.
