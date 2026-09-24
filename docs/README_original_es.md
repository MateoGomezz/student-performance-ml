# Proyecto Final — Machine Learning
### Predicción y análisis del rendimiento académico de estudiantes

---

## Lo más importante antes de empezar

Este proyecto no es solo código que funciona — es un **análisis profundo y razonado** de los resultados. El verdadero valor del trabajo está en:

- **Los notebooks de `resumenes/`**: contienen conclusiones extendidas, visualizaciones comentadas y razonamientos detallados que van mucho más allá de las métricas. Leerlos es indispensable para entender qué se ha hecho y por qué.
- **`resumenes/decisiones_disenyo.ipynb`**: recoge todas las decisiones de diseño de cada modelo (por qué ese preprocesado, por qué ese rango de hiperparámetros, qué se descartó y por qué).
- La memoria PDF (`memorias/Memoria_Final.pdf`) contiene las conclusiones globales del proyecto junto con versiones compactas de los resultados de cada sección.

---

## Dataset

**`rendimiento_estudiantes.csv`** — Dataset con variables sociodemográficas, académicas y de comportamiento de estudiantes universitarios portugueses.

- **Target de clasificación:** `objetivo` — situación académica final en 3 clases:
  - `abandono` — el estudiante abandonó la carrera
  - `graduado` — el estudiante finalizó y se graduó
  - `matriculado` — el estudiante sigue matriculado (clase de tránsito)
- **Target de regresión:** `nota_media_2sem` — nota media del 2º semestre (variable continua)

---

## Flujo de ejecución

> **IMPORTANTE:** `preprocessing.py` debe ejecutarse **primero y obligatoriamente** antes que cualquier modelo. Los modelos leen los ficheros generados por el preprocesado.

### Paso 0 — Preprocesado (ejecutar primero)

```bash
python preprocessing.py
```

Genera automáticamente la carpeta `data_processed/` con los ficheros listos para modelado:

| Fichero | Uso |
|---|---|
| `X_train.csv` / `X_test.csv` | Features completas (clasificación + no supervisado) |
| `X_train_reg.csv` / `X_test_reg.csv` | Features sin columnas del 2º semestre (regresión lineal) |
| `y_train.csv` / `y_test.csv` | Targets: `objetivo` + `nota_media_2sem` |

**Pipeline aplicado:** agrupación semántica de variables de alta cardinalidad → Label Encoding → One-Hot Encoding → transformación Yeo-Johnson (|skew| > 1) → split estratificado 70/30 → StandardScaler (fit solo en train).

> Ver análisis completo en `resumenes/resumen_preprocesado.ipynb`

---

### Paso 1 — Modelos de clasificación supervisada

Cada script carga los datos de `data_processed/`, realiza GridSearchCV con validación cruzada k=5, reentrena el mejor modelo sobre todo el train y evalúa sobre test. Las figuras se guardan automáticamente en `figures/`.

#### KNN — K-Nearest Neighbors
```bash
python models/knn.py
```
- GridSearch: 8 valores de K × 2 métricas de distancia = 80 entrenamientos
- Salidas: curva K vs F1, matriz de confusión, métricas por clase, análisis de distancias
- **Conclusiones en:** `resumenes/resumen_knn.ipynb`

#### Regresión Logística
```bash
python models/logistic_regression.py
```
- GridSearch: 6 valores de C × 2 regularizaciones (L1/L2) = 60 entrenamientos
- Salidas: curva C vs F1, matriz de confusión, top features por clase, análisis de colinealidad
- **Conclusiones en:** `resumenes/resumen_regresion_logistica.ipynb`

#### Random Forest
```bash
python models/random_forest.py
```
- GridSearch: 2×2×2 combinaciones × 5 folds = 40 entrenamientos
- Salidas: heatmap CV, matriz de confusión, feature importance
- **Conclusiones en:** `resumenes/resumen_random_forest.ipynb`

---

### Paso 2 — Regresión lineal (predicción continua)

```bash
python models/linear_regression.py
```
- Predice `nota_media_2sem` usando features sin columnas del 2º semestre (`X_train_reg`)
- GridSearch: Ridge vs Lasso × 6 alphas = 60 entrenamientos
- Métricas: RMSE y R²
- Salidas: curva alpha vs RMSE (CV), predicciones vs reales por clase, análisis de residuos, coeficientes Lasso
- **Conclusiones en:** `resumenes/resumen_regresion.ipynb`

---

### Paso 3 — Aprendizaje no supervisado

```bash
python models/unsupervised.py
```

Pipeline completo sin etiquetas:
1. PCA sobre 14 variables numéricas → scree plot → N componentes (80% varianza)
2. Visualización 2D (PC1 vs PC2) coloreada por etiqueta real
3. Análisis de loadings
4. Elección de K óptimo: método del codo + Silhouette
5. K-Means sobre el espacio PCA reducido
6. Heatmap de perfiles de clusters (valores originales)
7. Análisis de estabilidad ARI

> **Decisión clave:** K-Means se aplica sobre los N primeros componentes PCA (espacio reducido, sin correlaciones), no sobre las features originales.

Salidas en `figures/`: `01_scree_plot.png` … `07_stability_ari.png`

- **Conclusiones en:** `resumenes/resumen_no_supervisado.ipynb`

---

### Paso 4 — Análisis adicional: SMOTE

```bash
python models/smote.py
```
Estudio comparativo de estrategias de balanceo de clases: SMOTE (sobremuestreo sintético en espacio de features) vs `class_weight="balanced"` (ponderación interna). Responde a: *¿cambia el rendimiento final si equilibramos los datos antes de entrenar en lugar de ponderar los errores?*

- **Conclusiones en:** `resumenes/resumen_smote_adicional.ipynb`

---

## Notebooks de resumen — lectura recomendada

| Notebook | Contenido |
|---|---|
| `resumenes/resumen_preprocesado.ipynb` | Análisis exploratorio y decisiones de preprocesado |
| `resumenes/resumen_knn.ipynb` | Resultados KNN + conclusiones extendidas |
| `resumenes/resumen_regresion_logistica.ipynb` | Resultados Regresión Logística + conclusiones extendidas |
| `resumenes/resumen_random_forest.ipynb` | Resultados Random Forest + conclusiones extendidas |
| `resumenes/resumen_regresion.ipynb` | Resultados Regresión Lineal + conclusiones extendidas |
| `resumenes/resumen_no_supervisado.ipynb` | Resultados PCA + K-Means + conclusiones extendidas |
| `resumenes/resumen_smote_adicional.ipynb` | Análisis comparativo SMOTE vs class_weight |
| **`resumenes/decisiones_disenyo.ipynb`** | **Decisiones de diseño razonadas de todos los modelos** |

> Las conclusiones de los notebooks son significativamente más ricas y detalladas que las que aparecen en la memoria PDF. Para una evaluación completa del trabajo analítico, la lectura de los notebooks es imprescindible.

---

## Estructura del proyecto

```
proyecto_final_ml/
│
├── rendimiento_estudiantes.csv       # Dataset original
│
├── preprocessing.py                  # [EJECUTAR PRIMERO] Preprocesado completo
│
├── models/                           # Modelos de ML (ejecutar tras preprocessing.py)
│   ├── knn.py                        # Clasificación: KNN
│   ├── logistic_regression.py        # Clasificación: Regresión Logística
│   ├── random_forest.py              # Clasificación: Random Forest
│   ├── linear_regression.py          # Regresión: predicción nota continua
│   ├── smote.py                      # Análisis adicional: balanceo de clases
│   └── unsupervised.py               # No supervisado: PCA + K-Means
│
├── data_processed/                   # Generado automáticamente por preprocessing.py
│   ├── X_train.csv / X_test.csv
│   ├── X_train_reg.csv / X_test_reg.csv
│   └── y_train.csv / y_test.csv
│
├── figures/                          # Generado automáticamente por los modelos
│   ├── fig_knn.png
│   ├── fig_logistic_regression.png
│   ├── fig_colinealidad.png
│   ├── fig_random_forest.png
│   ├── fig_linear_regression.png
│   ├── fig_linear_regression_diagnostico.png
│   ├── fig_smote_comparativa.png
│   ├── fig_smote_metricas_clase.png
│   └── 01_scree_plot.png … 07_stability_ari.png
│
├── resumenes/                        # El análisis real está aquí
│   ├── decisiones_disenyo.ipynb
│   ├── resumen_preprocesado.ipynb
│   ├── resumen_knn.ipynb
│   ├── resumen_regresion_logistica.ipynb
│   ├── resumen_random_forest.ipynb
│   ├── resumen_regresion.ipynb
│   ├── resumen_no_supervisado.ipynb
│   └── resumen_smote_adicional.ipynb
│
└── memorias/
    └── Memoria_Final.pdf             # Memoria del proyecto (resumen compacto)
```

---

## Dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn scipy
```

Python 3.10+. Todos los scripts son autocontenidos y no requieren argumentos.
