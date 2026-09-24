"""
Clasificación multiclase con K-Nearest Neighbors (KNN).

Flujo:
    1. Cargar X_train, X_test, y_train, y_test
    2. GridSearchCV (k=5 folds) → 8 valores de K × 2 valores de p = 80 entrenamientos
    3. Reentrenar el mejor modelo sobre todo el train
    4. Evaluar sobre test → métricas finales
    5. Visualizar: curva K vs F1, matriz de confusión, métricas por clase, distancias vecinos
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
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
COLORES_CLASES = [AZUL, VERDE, ROJO]   # abandono, graduado, matriculado

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processed")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────────
"""

Cargamos los mismos ficheros que hemos usado en regresión logística ya que son
los que hemos preparado rigurosamente para clasificación en preprocessing.py
El escalado hecho en la etapa de preprocesamiento es clave para este modelo.
"""

print("[ 1/5 ] Cargando datos...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv")

y_train_enc = y_train["objetivo_encoded"].values
y_test_enc  = y_test["objetivo_encoded"].values
clases      = ["abandono", "graduado", "matriculado"]   # orden alfabético = 0,1,2

print(f"        X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"        Clases (train): { {c: (y_train['objetivo']==c).sum() for c in clases} }")

# ── 2. GRIDSEARCHCV ───────────────────────────────────────────────────────────
"""
Aquí, vamos a encontrar los hiperparámetros clave en KNN que son:
--> K = número de vecinos
Usamos valores con escala logarítmica para cubrir un buen abanico
--> p = parámetro de la distancia de Minkowski
Usaremos p= 1 que es distancia Manhattan y p= 2 que es la distancia
Euclídea

También, usamos weights = "distance" para que cada vecino que vota
sea ponderado por la inversa de su distancia. Así un vecino más
cercano tiene más influencia que uno lejano. Con esto reducimos un
poco el efecto del desbalanceo de clases.

Tenemos 8 valores de K x 5 folds = 80 entrenamientos
Usaremos como métrica de selección Macro F1-score por la misma razón
que en regresión logística.

"""

print("\n[ 2/5 ] GridSearchCV (k=5, 16 combinaciones = 80 entrenamientos)...")

param_grid = {
    "n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31],
    "p":           [1, 2],
}

base_model = KNeighborsClassifier(
    weights="distance",   # pondera voto por inversa de distancia
    algorithm="auto",     # sklearn elige la estructura más eficiente (KD-tree, ball-tree...)
    n_jobs=-1,            # usa todos los cores disponibles para el cálculo de distancias
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
"""
GridSearchCV con refit=True (por defecto) ya ha reentrenado el mejor modelo
sobre todo X_train. Lo extraemos directamente.
"""
print("\n[ 3/5 ] Reentrenando mejor modelo sobre todo el train...")
mejor_modelo = grid_search.best_estimator_

# ── 4. EVALUACIÓN SOBRE TEST ──────────────────────────────────────────────────
"""
Evaluamos el modelo con sus hiperparámetros óptimos (K,p) sobre el conjunto de test
y obtenemos las métricas con las que luego analizaremos el rendimiento del modelo
"""
print("\n[ 4/5 ] Evaluando sobre test...")
y_pred = mejor_modelo.predict(X_test)

acc      = accuracy_score(y_test_enc, y_pred)
f1_macro = f1_score(y_test_enc, y_pred, average="macro")
cm       = confusion_matrix(y_test_enc, y_pred)
report   = classification_report(y_test_enc, y_pred,
                                  target_names=clases, output_dict=True)

best_k = grid_search.best_params_["n_neighbors"]
best_p = grid_search.best_params_["p"]
nombre_dist = "Manhattan" if best_p == 1 else "Euclídea"

print(f"\n  ── MÉTRICAS FINALES (TEST) ──────────────────────────")
print(f"  Accuracy:       {acc:.4f}")
print(f"  Macro F1-score: {f1_macro:.4f}  ← métrica principal")
print(f"  K óptimo:       {best_k} vecinos")
print(f"  p óptimo:       {best_p} ({nombre_dist})")
print(f"\n  Por clase:")
for cls in clases:
    r = report[cls]
    print(f"    {cls:15s}  precision={r['precision']:.3f}  "
          f"recall={r['recall']:.3f}  F1={r['f1-score']:.3f}  "
          f"n={r['support']:.0f}")

# ── 5. VISUALIZACIONES ────────────────────────────────────────────────────────
"""
Devolvemos un panel con cuatro visualizaciones que explicamos más en profunfidad
en nuestro notebook.
"""
print("\n[ 5/5 ] Generando visualizaciones...")

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 5a. Curva K vs Macro F1 (CV) ─────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])

cv_results = pd.DataFrame(grid_search.cv_results_)
k_vals = param_grid["n_neighbors"]

for p_val, color, label in [(1, AZUL, "p=1 (Manhattan)"), (2, ROJO, "p=2 (Euclídea)")]:
    subset = cv_results[cv_results["param_p"] == p_val].sort_values("param_n_neighbors")
    ax0.plot(range(len(k_vals)), subset["mean_test_score"].values,
             marker="o", color=color, label=label, linewidth=2)
    ax0.fill_between(
        range(len(k_vals)),
        subset["mean_test_score"] - subset["std_test_score"],
        subset["mean_test_score"] + subset["std_test_score"],
        alpha=0.15, color=color
    )

ax0.set_xticks(range(len(k_vals)))
ax0.set_xticklabels([str(k) for k in k_vals])
ax0.set_xlabel("Número de vecinos K")
ax0.set_ylabel("Macro F1-score (CV)")
ax0.set_title("Búsqueda de hiperparámetros — GridSearchCV", fontweight="bold")
ax0.legend(fontsize=9, frameon=False)
ax0.yaxis.grid(True, linestyle="--", alpha=0.4)
ax0.set_axisbelow(True)

# Marcar el óptimo
best_k_idx = k_vals.index(best_k)
ax0.axvline(best_k_idx, color="gray", linestyle=":", linewidth=1.2)
ax0.annotate(
    f"Mejor: K={best_k}\np={best_p}\nF1={grid_search.best_score_:.3f}",
    xy=(best_k_idx, grid_search.best_score_),
    xytext=(best_k_idx + 0.4, grid_search.best_score_ - 0.03),
    fontsize=8, color="gray",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
)

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
etiquetas_metricas = ["Precision", "Recall", "F1-score"]
x = np.arange(len(clases))
ancho = 0.25

for i, (met, label) in enumerate(zip(metricas, etiquetas_metricas)):
    vals = [report[c][met] for c in clases]
    bars = ax2.bar(x + i * ancho, vals, ancho, label=label,
                   color=[AZUL, VERDE, ROJO][i], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax2.set_xticks(x + ancho)
ax2.set_xticklabels(clases)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel("Valor")
ax2.set_title("Precision, Recall y F1 por clase", fontweight="bold")
ax2.legend(fontsize=9, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

# ── 5d. Distribución de distancia al K-ésimo vecino por clase ─────────────────
"""
Para cada muestra de test calculamos la distancia a su K-ésimo vecino más cercano
en train. Esto mide la "confianza geométrica" del modelo: distancias pequeñas
indican que el punto vive en una zona densa y bien representada del espacio;
distancias grandes indican zonas escasas donde KNN es menos fiable.
Si matriculado tiene distancias sistemáticamente mayores, confirma que esa clase
ocupa una región más dispersa del espacio de features → más difícil de clasificar.
"""
ax3 = fig.add_subplot(gs[1, 1])

# kneighbors devuelve (distancias, índices) para cada punto de test
distancias_todos, _ = mejor_modelo.kneighbors(X_test)
dist_k_esimo = distancias_todos[:, -1]   # distancia al vecino más lejano (el K-ésimo)

for idx_clase, (clase, color) in enumerate(zip(clases, COLORES_CLASES)):
    mask = y_test_enc == idx_clase
    ax3.hist(dist_k_esimo[mask], bins=30, alpha=0.6,
             color=color, label=f"{clase} (n={mask.sum()})",
             edgecolor="white", linewidth=0.4)

ax3.set_xlabel(f"Distancia al vecino K={best_k} (distancia {nombre_dist})")
ax3.set_ylabel("Frecuencia")
ax3.set_title(
    f"Distribución de distancias al K-ésimo vecino por clase\n"
    f"(distancias grandes = zona poco densa = más incertidumbre)",
    fontweight="bold", fontsize=10
)
ax3.legend(fontsize=9, frameon=False)
ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)

# ── Título general ────────────────────────────────────────────────────────────
fig.suptitle(
    f"K-Nearest Neighbors — Mejor modelo: K={best_k}, p={best_p} ({nombre_dist})  |  "
    f"Accuracy={acc:.3f}  |  Macro F1={f1_macro:.3f}",
    fontsize=12, fontweight="bold", y=1.01
)

plt.savefig(f"{FIGURES_DIR}/fig_knn.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Guardado: {FIGURES_DIR}/fig_knn.png")
print("\n[ DONE ] KNN completado.")
