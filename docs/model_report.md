# Modelo de Análisis de Sentimiento — IMDB en Español

## Índice

1. [Descripción general](#descripción-general)
2. [Dataset](#dataset)
3. [Experimentos realizados](#experimentos-realizados)
4. [Modelo seleccionado: `baseline_tfidf_logreg`](#modelo-seleccionado-baseline_tfidf_logreg)
   - [Arquitectura](#arquitectura)
   - [Preprocesamiento](#preprocesamiento)
   - [Hiperparámetros](#hiperparámetros)
   - [Por qué es el mejor](#por-qué-es-el-mejor)
5. [Métricas de evaluación](#métricas-de-evaluación)
6. [API de inferencia](#api-de-inferencia)
   - [Endpoints](#endpoints)
   - [Ejemplos de uso](#ejemplos-de-uso)

---

## Descripción general

Este proyecto entrena y evalúa múltiples modelos de clasificación binaria de sentimiento (positivo / negativo) sobre reseñas de películas en español del dataset IMDB. Tras comparar ocho configuraciones distintas — baseline clásico, embeddings entrenables y embeddings preentrenados (spaCy, ConceptNet) — el mejor modelo se expone como servicio REST mediante FastAPI y se rastrea con MLflow.

---

## Dataset

| Atributo | Valor |
|---|---|
| Fuente | Kaggle — `luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish` |
| Idioma | Español (traducción automática del dataset original de IMDB) |
| Total de reseñas | 50 000 |
| Clases | `positive` / `negative` (balanceado: 50% cada una) |
| División entrenamiento | 40 000 (80%) |
| División validación | 5 000 (10%) |
| División test | 5 000 (10%) |

---

## Experimentos realizados

Se entrenaron y evaluaron ocho configuraciones con la misma división de datos y semilla aleatoria (42):

| Modelo | Familia | Cobertura embedding | F1 test | Accuracy test |
|---|---|---|---|---|
| **baseline_tfidf_logreg** | Baseline | N/A (sin embedding) | **0.8996** | **0.8996** |
| dense_trainable_embeddings | Neural | N/A (aprende desde cero) | 0.8806 | 0.8806 |
| conceptnet_finetuned | Neural | 32.62 % | 0.8804 | 0.8804 |
| spacy_lg_finetuned | Neural | 84.72 % | 0.8768 | 0.8768 |
| spacy_md_finetuned | Neural | 84.72 % | 0.8695 | 0.8695 |
| conceptnet_frozen | Neural | 32.62 % | 0.8014 | 0.8014 |
| spacy_md_frozen | Neural | 84.72 % | 0.7920 | 0.7920 |
| spacy_lg_frozen | Neural | 84.72 % | 0.7905 | 0.7912 |

Métrica principal de selección: **F1 macro promedio sobre el conjunto de test**.

---

## Modelo seleccionado: `baseline_tfidf_logreg`

**ID MLflow:** `m-414b9c0dea8b42c1a3c651845cb6e568`  
**Run ID:** `c7b6051007814f5fb79fb9d985e9faff`  
**Creado:** 2026-04-12 16:58 UTC

### Arquitectura

El modelo es un pipeline secuencial de scikit-learn compuesto por dos etapas:

```
Texto en español
      │
      ▼
┌─────────────────────────────────────────────┐
│  1. TfidfVectorizer                         │
│     · Preprocesamiento: normalize_text()    │
│     · max_features = 50 000                 │
│     · ngram_range  = (1, 2)  ← unigramas   │
│                                 y bigramas  │
│     · min_df = 2  ← ignora tokens          │
│                     que aparecen < 2 veces  │
│     · Salida: vector disperso TF-IDF        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  2. LogisticRegression                      │
│     · C = 4.0  (regularización L2 suave)   │
│     · solver = liblinear                   │
│     · max_iter = 2 000                     │
│     · Salida: P(positivo) ∈ [0, 1]         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
       umbral = 0.5
       ├── P ≥ 0.5 → "positive"
       └── P < 0.5 → "negative"
```

### Preprocesamiento

La función `normalize_text()` se aplica como `preprocessor` del `TfidfVectorizer`:

1. `html.unescape()` — decodifica entidades HTML (`&amp;`, `&lt;`, etc.)
2. Conversión a minúsculas
3. Eliminación de saltos de línea HTML (`<br />`)
4. Eliminación de etiquetas HTML restantes (`<b>`, `<i>`, etc.)
5. Sustitución de signos de puntuación por espacios
6. Colapso de espacios múltiples
7. Strip del resultado

### Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| `max_features` | 50 000 | Captura el vocabulario relevante sin ruido de términos muy raros |
| `ngram_range` | (1, 2) | Los bigramas capturan negaciones (`no me gustó`) y frases de opinión (`muy buena`) |
| `min_df` | 2 | Elimina hapax legomena que no generalizan |
| `C` (LR) | 4.0 | Regularización moderada; validado empíricamente |
| `solver` | liblinear | Eficiente para matrices dispersas de alta dimensión |
| `threshold` | 0.5 | Umbral de clasificación; clases balanceadas al 50% |

### Por qué es el mejor

**1. Cobertura total del vocabulario**

TF-IDF construye su diccionario directamente desde el corpus de entrenamiento: no hay tokens fuera de vocabulario (OOV). Los embeddings preentrenados solo cubren entre el **32 %** (ConceptNet en español) y el **85 %** (spaCy `es_core_news_lg`) del vocabulario real del dataset. Los tokens sin vector reciben representación cero, perdiendo información discriminativa.

**2. La tarea es léxicamente explícita**

Las reseñas de películas expresan opinión con vocabulario altamente polarizado:  
`excelente`, `magnífica`, `horrible`, `malísima`, `aburrida`, `impresionante`.  
Los n-gramas capturan exactamente esta señal. La semántica distribucional y contextual que aportan los embeddings no añade ventaja para este dominio.

**3. Los embeddings congelados penalizan fuertemente**

Cuando los pesos del embedding son fijos y la cobertura es baja, el modelo no puede compensar los tokens sin representación durante el entrenamiento. La diferencia entre frozen y finetuned es de **8–10 puntos de F1** en spaCy:

| spaCy lg frozen | 0.7905 | → | spaCy lg finetuned | 0.8768 |

Aun afinando, ningún modelo neuronal supera al baseline.

**4. Ventajas operacionales**

- **Sin dependencia de TensorFlow/GPU** — el modelo sklearn corre en cualquier entorno
- **Inferencia determinista y rápida** — sin batch sizes, sin hardware especializado
- **Trazabilidad total** — los coeficientes de la regresión logística son interpretables

---

## Métricas de evaluación

### Conjunto de test (5 000 reseñas, nunca vistas durante entrenamiento)

| Métrica | Macro avg | Clase: negative | Clase: positive |
|---|---|---|---|
| Accuracy | 0.8996 | — | — |
| F1-score | 0.8996 | 0.8987 | 0.9004 |
| Precision | 0.8997 | 0.9064 | 0.8930 |
| Recall | 0.8996 | 0.8912 | 0.9080 |
| Support | 5 000 | 2 500 | 2 500 |

### Conjunto de validación (5 000 reseñas, usadas solo para monitoreo)

| Métrica | Valor |
|---|---|
| Accuracy | 0.8968 |
| F1-score (macro) | 0.8968 |

La diferencia val/test es menor a **0.003**, lo que indica ausencia de sobreajuste al conjunto de validación.

---

## API de inferencia

La API está construida con **FastAPI** y sirve el modelo MLflow (`SentimentPyfuncModel`) cargado en memoria al iniciar.

**URL base:** `http://<host>:8000`  
**Documentación interactiva:** `http://<host>:8000/docs`

### Endpoints

#### `GET /health`

Verifica que el servicio está activo y devuelve la ruta del modelo cargado.

**Respuesta:**
```json
{
  "status": "ok",
  "model_uri": "/ruta/al/models/champion"
}
```

---

#### `GET /model-info`

Devuelve la ficha técnica completa del modelo: métricas, criterios de selección, comparación de experimentos y metadatos MLflow.

**Respuesta (extracto):**
```json
{
  "model_id": "m-414b9c0dea8b42c1a3c651845cb6e568",
  "model_name": "baseline_tfidf_logreg",
  "backend": "sklearn",
  "threshold": 0.5,
  "mlflow_run_id": "c7b6051007814f5fb79fb9d985e9faff",
  "test_metrics": {
    "accuracy": 0.8996,
    "f1": 0.8996,
    "precision": 0.8997,
    "recall": 0.8996
  },
  "selection_criteria": {
    "metric": "test_f1 (macro avg)",
    "rationale": ["TF-IDF cubre el 100% del vocabulario...", "..."]
  },
  "experiment_comparison": [
    {"model": "baseline_tfidf_logreg", "test_f1": 0.8996, "selected": true},
    ...
  ]
}
```

---

#### `POST /predict`

Clasifica una o varias reseñas en español.

**Request body:**
```json
{
  "texts": [
    "Una película absolutamente espectacular, la mejor del año.",
    "Guión pésimo, actuaciones mediocres, una pérdida de tiempo."
  ]
}
```

**Respuesta:**
```json
{
  "predictions": [
    {
      "text": "Una película absolutamente espectacular, la mejor del año.",
      "probability": 0.9991,
      "predicted_class": 1,
      "predicted_label": "positive"
    },
    {
      "text": "Guión pésimo, actuaciones mediocres, una pérdida de tiempo.",
      "probability": 0.0021,
      "predicted_class": 0,
      "predicted_label": "negative"
    }
  ]
}
```

**Campos de respuesta:**

| Campo | Tipo | Descripción |
|---|---|---|
| `text` | string | Texto original de entrada |
| `probability` | float [0, 1] | Probabilidad de clase positiva |
| `predicted_class` | int (0 ó 1) | 0 = negativo, 1 = positivo |
| `predicted_label` | string | `"positive"` o `"negative"` |

### Ejemplos de uso

**curl:**
```bash
curl -X POST http://<host>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Excelente película, la recomiendo totalmente."]}'
```

**Python:**
```python
import requests

response = requests.post(
    "http://<host>:8000/predict",
    json={"texts": [
        "Excelente película, la recomiendo totalmente.",
        "Muy aburrida, no la vean.",
    ]},
)
print(response.json())
```

**Arrancar el servicio:**
```bash
# Con el modelo campeón por defecto (models/champion/)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Con un modelo específico
MODEL_URI=/ruta/al/modelo uvicorn api.main:app --host 0.0.0.0 --port 8000

# Usando Make
make api
```
