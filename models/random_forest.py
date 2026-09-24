"""
Clasificación multiclase con Random Forest.

Flujo:
    1. Cargar X_train, X_test, y_train, y_test
    2. GridSearchCV (k=5 folds) → 2×2×2 = 8 combinaciones × 5 folds = 40 entrenamientos
    3. Reentrenar el mejor modelo sobre todo el train
    4. Evaluar sobre test → métricas finales
    5. Visualizar: heatmap CV, matriz de confusión, métricas por clase, feature importance
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Estilo global ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
})
AZUL   = "#2E75B6"
ROJO   = "#E8593C"
VERDE  = "#2E8B57"
COLORES_CLASES = [AZUL, VERDE, ROJO]

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processed")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────────
"""
Usamos los mismos ficheros que para los otros modelos de clasificación. Sin embargo,
Random Forest no requiere escalado ya que no trabaja con distancias pero como
lo tenemos hecho ya desde el preprocesado y no perjudica al modelo pues trabajaremos
con los datos escalados.
"""
print("[ 1/5 ] Cargando datos...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv")

y_train_enc = y_train["objetivo_encoded"].values
y_test_enc  = y_test["objetivo_encoded"].values
clases      = ["abandono", "graduado", "matriculado"]

print(f"        X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"        Clases (train): { {c: (y_train['objetivo']==c).sum() for c in clases} }")

# ── 2. GRIDSEARCHCV ───────────────────────────────────────────────────────────
"""
Vamos a buscar los tres hiperparámetros clave en Random Forest:
--> n_estimators: número de árboles del ensemble, probamos entre dos valores que
son 100 y 300. Más árboles siempre reduce varianza.

--> max_features: define las features candidatas por cada split. Es nuestro mecanismo
central para la decorrelación entre variables en los árboles. Probamos entre
"sqrt" = √98 ≈ 10 features o 20 que produce menos decorrelación pero más info
por split.

--> max_depth: Probamos entre "None" = sin límite, dejamos los árboles crecer hasta
hojas puras produciendo overfitting individual de los árboles o "20" que genera
árboles profundos pero regularizados

Total: 2 × 2 × 2 × 5 folds = 40 entrenamientos.
Métrica: Macro F1-score, misma razón que en modelos anteriores.
"""
print("\n[ 2/5 ] GridSearchCV (k=5, 8 combinaciones = 40 entrenamientos)...")

param_grid = {
    "n_estimators": [100, 300],
    "max_features": ["sqrt", 20],
    "max_depth":    [None, 20],
}

base_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train_enc)

print(f"\n        Mejor combinación: {grid_search.best_params_}")
print(f"        Mejor Macro F1 (CV): {grid_search.best_score_:.4f}")

# ── 3. REENTRENAR SOBRE TODO EL TRAIN ─────────────────────────────────────────
print("\n[ 3/5 ] Reentrenando mejor modelo sobre todo el train...")
mejor_modelo = grid_search.best_estimator_
best_n  = grid_search.best_params_["n_estimators"]
best_mf = grid_search.best_params_["max_features"]
best_md = grid_search.best_params_["max_depth"]

# ── 4. EVALUACIÓN SOBRE TEST ──────────────────────────────────────────────────
print("\n[ 4/5 ] Evaluando sobre test...")
y_pred   = mejor_modelo.predict(X_test)
acc      = accuracy_score(y_test_enc, y_pred)
f1_macro = f1_score(y_test_enc, y_pred, average="macro")
cm       = confusion_matrix(y_test_enc, y_pred)
report   = classification_report(y_test_enc, y_pred,
                                  target_names=clases, output_dict=True)

print(f"\n  ── MÉTRICAS FINALES (TEST) ──────────────────────────")
print(f"  Accuracy:       {acc:.4f}")
print(f"  Macro F1-score: {f1_macro:.4f}  ← métrica principal")
print(f"  n_estimators:   {best_n} | max_features: {best_mf} | max_depth: {best_md}")
print(f"\n  Por clase:")
for cls in clases:
    r = report[cls]
    print(f"    {cls:15s}  precision={r['precision']:.3f}  "
          f"recall={r['recall']:.3f}  F1={r['f1-score']:.3f}  "
          f"n={r['support']:.0f}")

# ── 5. VISUALIZACIONES ────────────────────────────────────────────────────────
print("\n[ 5/5 ] Generando visualizaciones...")

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 5a. Heatmap CV: max_features × max_depth ─────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])

cv_results = pd.DataFrame(grid_search.cv_results_)
subset = cv_results[cv_results["param_n_estimators"] == best_n].copy()
subset["md_label"] = subset["param_max_depth"].apply(
    lambda x: "None" if x is None or str(x) == "None" else str(x))
subset["mf_label"] = subset["param_max_features"].astype(str)

pivot = subset.pivot_table(
    index="mf_label", columns="md_label", values="mean_test_score"
)
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=ax0, cbar=True, linewidths=0.5,
            annot_kws={"size": 11, "weight": "bold"})
ax0.set_title(
    f"Macro F1 (CV) por hiperparámetros\n(n_estimators={best_n} — valor óptimo)",
    fontweight="bold")
ax0.set_xlabel("max_depth")
ax0.set_ylabel("max_features")

# ── 5b. Matriz de confusión ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=clases, yticklabels=clases,
            ax=ax1, cbar=True, linewidths=0.5)
ax1.set_title("Matriz de confusión (normalizada por fila)", fontweight="bold")
ax1.set_xlabel("Predicción")
ax1.set_ylabel("Real")
for i in range(len(clases)):
    for j in range(len(clases)):
        ax1.text(j + 0.5, i + 0.75, f"n={cm[i,j]}",
                 ha="center", va="center", fontsize=7, color="gray")

# ── 5c. Métricas por clase ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

metricas = ["precision", "recall", "f1-score"]
x = np.arange(len(clases))
ancho = 0.25
for i, (met, label) in enumerate(zip(metricas, ["Precision", "Recall", "F1-score"])):
    vals = [report[c][met] for c in clases]
    bars = ax2.bar(x + i*ancho, vals, ancho, label=label,
                   color=[AZUL, VERDE, ROJO][i], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax2.set_xticks(x + ancho)
ax2.set_xticklabels(clases)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel("Valor")
ax2.set_title("Precision, Recall y F1 por clase", fontweight="bold")
ax2.legend(fontsize=9, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

# ── 5d. Feature importance top 15 ────────────────────────────────────────────
"""
Importancia calculada como reducción media de impureza Gini en todos los splits
de todos los árboles donde aparece cada feature. Es siempre positiva (mide
relevancia, no dirección) y no lineal (captura interacciones entre variables).
Permite comparar con los coeficientes de regresión logística: esperamos que
las features más importantes aquí coincidan con las de mayor peso allí,
validando la coherencia entre un modelo lineal y uno de ensemble no lineal.
"""
ax3 = fig.add_subplot(gs[1, 1])

feature_names = X_train.columns.tolist()
importances   = mejor_modelo.feature_importances_
sorted_idx    = np.argsort(importances)[::-1][:15]
top_names     = [feature_names[i] for i in sorted_idx]
top_vals      = importances[sorted_idx]
colores_imp   = [ROJO if v >= np.median(top_vals) else AZUL for v in top_vals]

ax3.barh(range(15), top_vals[::-1], color=colores_imp[::-1],
         alpha=0.85, edgecolor="white")
ax3.set_yticks(range(15))
ax3.set_yticklabels(top_names[::-1], fontsize=8)
ax3.set_xlabel("Importancia media (reducción de impureza Gini)")
ax3.set_title(
    "Top 15 features — Importancia global del modelo\n"
    "(rojo = más importante, azul = menos importante)",
    fontweight="bold", fontsize=10)
ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)

# ── Título general ────────────────────────────────────────────────────────────
fig.suptitle(
    f"Random Forest — n_estimators={best_n}, max_features={best_mf}, "
    f"max_depth={best_md}  |  Accuracy={acc:.3f}  |  Macro F1={f1_macro:.3f}",
    fontsize=12, fontweight="bold", y=1.01
)

plt.savefig(f"{FIGURES_DIR}/fig_random_forest.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Guardado: {FIGURES_DIR}/fig_random_forest.png")
print("\n[ DONE ] Random Forest completado.")
