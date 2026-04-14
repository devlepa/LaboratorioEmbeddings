# IMDB Spanish Sentiment — Laboratorio de Embeddings

Análisis de sentimientos en español sobre reseñas de cine (IMDB). Compara 8 experimentos —un baseline clásico contra redes neuronales con distintos embeddings— registra todo en MLflow y expone el modelo ganador via FastAPI.

**Modelo en producción:** `baseline_tfidf_logreg` (TF-IDF + Regresión Logística)  
**F1 macro (test):** 0.8996 &nbsp;|&nbsp; **ROC-AUC:** 0.9626

Servicios desplegados en EC2:
- MLflow UI: `http://ec2-13-223-188-95.compute-1.amazonaws.com:5000`
- API REST:  `http://ec2-13-223-188-95.compute-1.amazonaws.com:8000`

---

## Tabla de contenidos

1. [Dataset y preparación de datos](#1-dataset-y-preparación-de-datos)
2. [Preprocesamiento de texto](#2-preprocesamiento-de-texto)
3. [Modelo baseline — TF-IDF + Regresión Logística](#3-modelo-baseline--tf-idf--regresión-logística)
4. [Modelo neuronal con embeddings entrenables](#4-modelo-neuronal-con-embeddings-entrenables)
5. [Modelos con embeddings preentrenados](#5-modelos-con-embeddings-preentrenados)
6. [Registro de experimentos en MLflow](#6-registro-de-experimentos-en-mlflow)
7. [Análisis comparativo](#7-análisis-comparativo)
8. [Despliegue](#8-despliegue)
9. [Estructura de archivos](#estructura-de-archivos)
10. [Setup y comandos](#setup-y-comandos)

---

## 1. Dataset y preparación de datos

Se utilizó el dataset **IMDB Dataset of 50K Movie Reviews (Spanish)** disponible en Kaggle (`luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish`), que contiene 50.000 reseñas de películas en español perfectamente balanceadas: 25.000 positivas y 25.000 negativas. Este balance eliminó la necesidad de técnicas de sobremuestreo o pesos de clase.

La descarga se maneja automáticamente con `kagglehub`, que cachea el dataset en `~/.cache/kagglehub/` para no volver a descargarlo en ejecuciones posteriores.

Las etiquetas `"positive"` / `"negative"` se mapearon a `1` / `0`. Los datos se dividieron en tres conjuntos usando división estratificada (preservando la proporción de clases):

| Conjunto | Proporción | Reseñas | Positivas | Negativas |
|----------|-----------|---------|-----------|-----------|
| Entrenamiento | 80% | 40.000 | 20.000 | 20.000 |
| Validación | 10% | 5.000 | 2.500 | 2.500 |
| Test | 10% | 5.000 | 2.500 | 2.500 |

La semilla fija (`random_state=42`) garantiza reproducibilidad en todos los experimentos.

---

## 2. Preprocesamiento de texto

Se implementaron **dos versiones equivalentes** del mismo pipeline de limpieza, una para cada tipo de modelo:

### Versión Python — `normalize_text()`

Usada en el pipeline sklearn (baseline):

```
html.unescape → elimina tags <br> → elimina tags HTML
→ elimina puntuación (incluye ¿ ¡ " " « » … ´ `)
→ colapsa espacios múltiples → lower()
```

### Versión TensorFlow — `keras_standardization()`

Usada como función `standardize=` dentro de `keras.layers.TextVectorization` para todos los modelos neuronales. Aplica exactamente la misma lógica pero operando sobre tensores de strings con `tf.strings.*`, lo que permite que el preprocesamiento ocurra **dentro del grafo de cómputo de Keras**, sin necesidad de preprocesar el texto antes de pasarlo al modelo.

```python
@keras.utils.register_keras_serializable(package='imdb_sentiment')
def keras_standardization(inputs: tf.Tensor) -> tf.Tensor:
    cleaned = tf.strings.lower(inputs)
    cleaned = tf.strings.regex_replace(cleaned, r'<br\s*/?>', ' ')
    cleaned = tf.strings.regex_replace(cleaned, r'<[^>]+>', ' ')
    cleaned = tf.strings.regex_replace(cleaned, r'&[a-zA-Z#0-9]+;', ' ')
    cleaned = tf.strings.regex_replace(cleaned, PUNCTUATION_PATTERN, ' ')
    cleaned = tf.strings.regex_replace(cleaned, r'\s+', ' ')
    return tf.strings.strip(cleaned)
```

La capa `TextVectorization` se adapta (`adapt()`) sobre el corpus de entrenamiento para construir el vocabulario, con un máximo de **30.000 tokens** y longitud de secuencia fija de **300 tokens**.

---

## 3. Modelo Baseline — TF-IDF + Regresión Logística

El baseline es un pipeline de sklearn con dos pasos:

### TF-IDF Vectorizer

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `max_features` | 50.000 | Cubre el vocabulario relevante sin incluir tokens extremadamente raros |
| `ngram_range` | (1, 2) | Los bigramas capturan expresiones de opinión compuestas (*muy bueno*, *no recomendable*) |
| `min_df` | 2 | Descarta hápax que solo generan ruido |
| `preprocessor` | `normalize_text` | Aplica el mismo pipeline de limpieza definido en `preprocessing.py` |

### Regresión Logística

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `C` | 4.0 | Regularización L2 moderada; permite cierta flexibilidad sin sobreajuste |
| `solver` | `liblinear` | Eficiente para vocabularios grandes y matrices dispersas |
| `max_iter` | 2000 | Suficiente para convergencia sobre 50k documentos |

### Resultados

| Métrica | Test | Validación |
|---------|------|-----------|
| Accuracy | 89.96% | 89.68% |
| F1 Macro | 89.96% | 89.68% |
| Precisión Macro | 89.97% | 89.68% |
| Recall Macro | 89.96% | 89.68% |
| ROC-AUC | 96.26% | — |

**Por clase (test):**

| Clase | Precisión | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negativo | 0.9064 | 0.8912 | 0.8987 | 2.500 |
| Positivo | 0.8930 | 0.9080 | 0.9004 | 2.500 |

---

## 4. Modelo neuronal con embeddings entrenables

Se diseñó una **arquitectura densa común** que se reutiliza en los 7 experimentos neuronales. Solo varía la capa de embeddings entre experimentos.

### Arquitectura

```
Input (texto raw, dtype=string)
    │
    ▼
TextVectorization (keras_standardization, max_tokens=30000, seq_len=300)
    │
    ▼
Embedding (vocab_size × embedding_dim, mask_zero=True)
    │
    ▼
GlobalAveragePooling1D
    │
    ▼
Dense(128, activation='relu') → Dropout(0.30)
    │
    ▼
Dense(64, activation='relu') → Dropout(0.30)
    │
    ▼
Dense(1, activation='sigmoid')  →  probabilidad ∈ [0, 1]
```

### Decisiones de diseño y justificaciones

| Hiperparámetro | Valor | Justificación |
|---------------|-------|---------------|
| `embedding_dim` | 128 | Equilibrio entre capacidad representacional y costo computacional. Para clasificación binaria de textos de longitud moderada, 128 dimensiones son suficientes sin sobreajuste. |
| `hidden_units` | [128, 64] | Dos capas con reducción progresiva (embudo) permiten aprender representaciones jerárquicas mientras el número de parámetros se mantiene manejable. |
| `dropout_rate` | 0.30 | Regularización suficiente para prevenir sobreajuste dado el tamaño del dataset (40k ejemplos de entrenamiento). |
| `sequence_length` | 300 | Las reseñas de IMDB son típicamente largas; 300 tokens capturan la mayor parte del contenido semántico relevante sin truncar demasiado. |
| `max_tokens` | 30.000 | Cubre el vocabulario de reseñas en español sin incluir términos extremadamente raros que solo generan ruido. |
| `GlobalAveragePooling1D` | — | Promedia los embeddings de todos los tokens. Es más robusto que `Flatten` o `GlobalMaxPooling` para textos de longitud variable, ya que no depende de un único token dominante. |

### Entrenamiento

```python
optimizer = Adam(learning_rate=0.001)
loss      = binary_crossentropy
batch_size = 64
epochs     = 12 (máximo)
callbacks  = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
```

`EarlyStopping` detiene el entrenamiento si `val_loss` no mejora en 3 épocas consecutivas y restaura automáticamente los mejores pesos, evitando sobreajuste.

### Resultados

**F1 macro (test): 0.8806**

---

## 5. Modelos con embeddings preentrenados

Se usaron **3 fuentes de embeddings preentrenados**, y para cada una se realizaron **2 experimentos** (congelado y ajustado), totalizando 6 experimentos. La arquitectura fue **idéntica** al punto anterior; solo se reemplazó la capa `Embedding`.

La función `build_embedding_matrix(vocabulary, config)` construye la matriz de pesos `(vocab_size, embedding_dim)` iterando sobre el vocabulario del `TextVectorization` y buscando el vector correspondiente en la fuente preentrenada. Los tokens sin vector reciben el vector cero.

### Fuente 1: spaCy `es_core_news_md`

- Modelo de spaCy entrenado sobre noticias en español (versión medium)
- Dimensión de embeddings: **96 dimensiones**
- Cobertura sobre el vocabulario del dataset: **~84.72%**
- Lookup: busca por token exacto, luego en minúsculas
- El ~15% sin cobertura corresponde a términos coloquiales, anglicismos y nombres propios frecuentes en reseñas de cine

### Fuente 2: spaCy `es_core_news_lg`

- Igual que `md` pero versión large con vocabulario más amplio
- Dimensión de embeddings: **300 dimensiones**
- Cobertura: **~84.72%** (prácticamente idéntica a la versión medium)
- La diferencia principal respecto a `md` es la dimensionalidad, no la cobertura

### Fuente 3: ConceptNet Numberbatch 17.06 (via Gensim)

- Grafo de conocimiento multilingüe con representaciones distribuidas
- Dimensión de embeddings: **300 dimensiones**
- Lookup con estilo `conceptnet_es`: busca como `/c/es/<token>`, luego como `<token>` normalizado, luego como `<token>` original
- Cobertura: **~32.62%** — el vocabulario informal de reseñas tiene muchos términos fuera del grafo de conocimiento formal

### Variantes por fuente: frozen vs. finetuned

| Variante | `trainable` | Comportamiento |
|----------|------------|----------------|
| **Frozen** | `False` | Los pesos del embedding NO se actualizan durante backpropagation. El modelo solo aprende a combinar representaciones fijas preentrenadas. |
| **Finetuned** | `True` | Los pesos se inicializan con los vectores preentrenados y se ajustan durante el entrenamiento. El modelo adapta las representaciones al dominio específico (reseñas de películas). |

### Resumen de los 6 experimentos con embeddings preentrenados

| Experimento | Fuente | Trainable | Dim | Cobertura | F1 (test) |
|-------------|--------|-----------|-----|-----------|-----------|
| `spacy_md_frozen` | spaCy md | No | 96 | 84.72% | 0.7920 |
| `spacy_md_finetuned` | spaCy md | Sí | 96 | 84.72% | 0.8695 |
| `spacy_lg_frozen` | spaCy lg | No | 300 | 84.72% | 0.7905 |
| `spacy_lg_finetuned` | spaCy lg | Sí | 300 | 84.72% | 0.8768 |
| `conceptnet_frozen` | ConceptNet | No | 300 | 32.62% | 0.8014 |
| `conceptnet_finetuned` | ConceptNet | Sí | 300 | 32.62% | 0.8804 |

---

## 6. Registro de experimentos en MLflow

Todos los experimentos se registraron en un servidor MLflow dedicado desplegado en EC2, accesible en `http://ec2-13-223-188-95.compute-1.amazonaws.com:5000`.

**Total: 8 runs** registradas bajo el experimento `imdb-spanish-sentiment`.

> El enunciado pedía 7; se implementaron 8 porque la fórmula correcta es:
> 1 baseline + 1 entrenable + (3 fuentes × 2 variantes) = **1 + 1 + 6 = 8**.
> Si se requieren exactamente 7, basta con comentar uno en `configs/experiments.yaml`.

### Contenido de cada run

**Parámetros registrados:**
- Nombre y tipo del experimento (`baseline` / `neural`)
- Fuente de embeddings, `trainable`, dimensión, cobertura
- Todos los hiperparámetros del modelo (según tipo)

**Métricas registradas** (en validación y en test):
- `accuracy`, `f1`, `precision`, `recall`, `roc_auc`

**Artefactos registrados:**

| Archivo | Descripción |
|---------|-------------|
| `analysis/test_classification_report.json` | Reporte completo por clase (precisión, recall, F1) |
| `analysis/validation_classification_report.json` | Ídem en validation set |
| `analysis/dataset_metadata.json` | Fuente, número de filas, distribución de clases |
| `analysis/experiment_config.json` | Nombre y tipo del experimento |
| `analysis/test_confusion_matrix.png` | Matriz de confusión sobre test set |
| `analysis/test_roc_curve.png` | Curva ROC sobre test set |
| `analysis/test_predictions.csv` | Probabilidades y predicciones por muestra |
| `model/` | Modelo serializado + metadata.json + código fuente |

### Wrapper de inferencia unificado — `SentimentPyfuncModel`

MLflow require un `PythonModel` para empaquetar el modelo con su lógica de inferencia. Se implementó `SentimentPyfuncModel(mlflow.pyfunc.PythonModel)` que unifica la interfaz de sklearn y Keras:

- `load_context()`: carga el `.joblib` (sklearn) o el `.keras` (Keras) según el campo `backend` en `metadata.json`
- `predict(df)`: acepta `DataFrame{"text": [...]}` y devuelve `[text, probability, predicted_class, predicted_label]`

Esto permite servir cualquier modelo desde la misma API REST sin cambios de código, independientemente del backend.

El modelo se sube a MLflow junto con el código fuente de `src/` (vía `code_paths`), de modo que el wrapper sea importable al servir el modelo desde cualquier entorno.

---

## 7. Análisis comparativo

### Resultados completos

| Modelo | F1 (test) | ROC-AUC | Cobertura vocab |
|--------|-----------|---------|-----------------|
| **baseline_tfidf_logreg** ★ | **0.8996** | **0.9626** | 100% |
| dense_trainable_embeddings | 0.8806 | — | 100% |
| conceptnet_finetuned | 0.8804 | 0.9493 | 32.62% |
| spacy_lg_finetuned | 0.8768 | — | 84.72% |
| spacy_md_finetuned | 0.8695 | — | 84.72% |
| conceptnet_frozen | 0.8014 | 0.8873 | 32.62% |
| spacy_md_frozen | 0.7920 | — | 84.72% |
| spacy_lg_frozen | 0.7905 | — | 84.72% |

### Por qué ganó el baseline

**1. Cobertura completa del vocabulario**

TF-IDF representa el 100% de los tokens del corpus. Los embeddings de spaCy dejan sin representar ~15% de los tokens (anglicismos, términos coloquiales de cine, neologismos); ConceptNet Numberbatch deja sin representar ~67%. Los tokens sin vector reciben el vector cero, lo que introduce ruido en la capa de embedding y degrada la capacidad del modelo para distinguir sentimientos.

**2. Naturaleza del texto**

Las reseñas de IMDB usan vocabulario explícitamente polarizado (*excelente*, *horrible*, *magnífico*, *decepcionante*). TF-IDF con bigramas captura directamente estas señales léxicas sin necesidad de representaciones distribuidas. Para este tipo de texto, la frecuencia ponderada de términos es una señal tan discriminativa como cualquier representación semántica densa.

**3. Impacto de congelar los embeddings**

Los modelos `_frozen` pierden hasta **~10 puntos de F1** respecto a sus versiones `_finetuned`. Los vectores preentrenados en noticias (spaCy) o grafo de conocimiento (ConceptNet) no están adaptados al dominio de reseñas de cine. Al congelarlos, el modelo no puede compensar esta inadaptación y la alta tasa de tokens sin vector deteriora fuertemente el rendimiento.

**4. El ajuste mejora pero no supera al baseline**

Los modelos `_finetuned` recuperan 7-9 puntos de F1 respecto a sus versiones `_frozen`, ajustando los vectores al dominio de reseñas. Sin embargo, siguen limitados porque el 15-67% de los tokens arranca del vector cero (sin información previa) y el entrenamiento es insuficiente para aprender representaciones útiles para todos ellos desde cero.

**5. ConceptNet ajustado supera a spaCy ajustado a pesar de menor cobertura**

ConceptNet finetuned (F1=0.8804) supera a spaCy lg finetuned (F1=0.8768) a pesar de tener menor cobertura (32% vs 85%). La explicación está en la **dimensionalidad**: ConceptNet usa vectores de 300 dimensiones contra las 96 de spaCy md. Al ajustar, la mayor capacidad representacional de ConceptNet compensa la cobertura baja.

**6. Complejidad sin ganancia**

La complejidad adicional de los modelos neuronales (entrenamiento iterativo, dependencia de GPU para eficiencia, mayor tiempo de inferencia) no se traduce en mejora de rendimiento en este corpus. El baseline es determinista, instantáneo en inferencia, no requiere TensorFlow y produce resultados perfectamente reproducibles.

### Limitaciones del baseline

El modelo TF-IDF + LogReg no captura contexto, negación ni ironía. Por ejemplo, la frase *"soy una perra muy linda y amo con mucho amor que rico comer chimbo el profe esta muy bueno"* (texto sin relación con cine) recibe probabilidad positiva de 73.5% porque contiene términos con carga léxica positiva en el vocabulario de entrenamiento (*linda*, *amo*, *bueno*). El modelo no sabe que el contexto es absurdo; solo detecta la frecuencia ponderada de señales léxicas. En dominios con ironía, sarcasmo o negación compleja, los modelos neuronales con embeddings contextuales serían más adecuados.

---

## 8. Despliegue

El modelo campeón (`baseline_tfidf_logreg`) se desplegó como una **API REST con FastAPI** en una instancia EC2, accesible en `http://ec2-13-223-188-95.compute-1.amazonaws.com:8000`.

### Diseño del despliegue

- El modelo se empaqueta en `models/champion/` (incluido en git) como un `mlflow.pyfunc` model. Esto elimina la dependencia del servidor MLflow en tiempo de arranque: con un `git pull` el modelo está disponible sin conexión al servidor de tracking.
- Al arrancar, la API carga el modelo **una sola vez** via `mlflow.pyfunc.load_model("models/champion")` usando el patrón `lifespan` de FastAPI.
- Los JSONs de análisis (métricas, configuración, comparación de experimentos) se leen desde `models/champion/artifacts/analysis/`, no desde `mlruns/` (que está en `.gitignore`). Si los JSONs locales no existen, la API hace fallback al servidor MLflow remoto via `MlflowClient`.

### Endpoints

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
  "test_metrics": {"accuracy": 0.8996, "f1": 0.8996, "roc_auc": 0.9626},
  "test_metrics_per_class": {
    "negative": {"precision": 0.9064, "recall": 0.8912, "f1": 0.8987, "support": 2500},
    "positive": {"precision": 0.8930, "recall": 0.9080, "f1": 0.9004, "support": 2500}
  },
  "validation_metrics": {...},
  "experiment_comparison": [...],
  "selection_criteria": {"metric": "test_f1 (macro avg)", "rationale": [...]}
}
```

#### `POST /predict`
```bash
curl -X POST http://ec2-13-223-188-95.compute-1.amazonaws.com:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Una película magnífica", "Terrible pérdida de tiempo"]}'
```
```json
{"predictions": [
  {"text": "Una película magnífica",       "probability": 0.923, "predicted_class": 1, "predicted_label": "positive"},
  {"text": "Terrible pérdida de tiempo",   "probability": 0.071, "predicted_class": 0, "predicted_label": "negative"}
]}
```

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
