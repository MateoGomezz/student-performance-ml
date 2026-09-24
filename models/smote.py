"""
Clasificación multiclase con SMOTE — Análisis comparativo de estrategias de balanceo.

SMOTE (Synthetic Minority Over-sampling TEchnique) genera ejemplos sintéticos de las
clases minoritarias interpolando entre muestras reales en el espacio de features.
A diferencia de class_weight="balanced" (que ajusta internamente los pesos del modelo),
SMOTE modifica directamente los datos de entrenamiento antes de entrenar el modelo.


¿Cambia el rendimiento final si equilibramos las clases antes de entrenar en lugar
de ponderar los errores durante el entrenamiento?

Flujo:
    1. Cargar X_train, X_test, y_train, y_test  (mismos ficheros que los modelos anteriores)
    2. Aplicar SMOTE sobre X_train / y_train  →  X_train_sm, y_train_sm
    3. GridSearchCV (k=5 folds) para Regresión Logística  →  misma rejilla que logistic_regression.py
    4. GridSearchCV (k=5 folds) para Random Forest        →  misma rejilla que random_forest.py
    5. Evaluar ambos modelos sobre test (sin tocar)
    6. Visualizar: distribución de clases antes/después de SMOTE, matrices de confusión,
       métricas por clase y tabla comparativa vs class_weight
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
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
NARANJA = "#FF9800"

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processed")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────────
"""
Usamos exactamente los mismos ficheros que en logistic_regression.py y random_forest.py.
El conjunto de test NO se toca en ningún momento: SMOTE solo se aplica sobre train.
"""
print("[ 1/6 ] Cargando datos...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv")

y_train_enc = y_train["objetivo_encoded"].values
y_test_enc  = y_test["objetivo_encoded"].values
clases      = ["abandono", "graduado", "matriculado"]   # orden alfabético = 0, 1, 2

print(f"        X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"        Distribución original (train):")
for i, c in enumerate(clases):
    n = (y_train_enc == i).sum()
    print(f"          {c:15s}: {n:4d}  ({100*n/len(y_train_enc):.1f}%)")

# ── 2. SMOTE ──────────────────────────────────────────────────────────────────
"""
SMOTE genera muestras sintéticas de las clases minoritarias interpolando entre
k vecinos más cercanos de la misma clase en el espacio de features.

Parámetros clave:
- sampling_strategy="auto": iguala todas las clases a la mayoritaria (graduado).
- k_neighbors=5: número de vecinos usados para generar cada muestra sintética.
  Es el valor por defecto y estándar en la literatura.
- random_state=42: reproducibilidad.

CRÍTICO: SMOTE se aplica SOLO sobre train. Aplicarlo sobre test contaminaría
la evaluación al introducir muestras sintéticas en el conjunto de generalización.
"""
print("\n[ 2/6 ] Aplicando SMOTE sobre train...")
smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train_enc)

print(f"        Distribución post-SMOTE (train):")
for i, c in enumerate(clases):
    n_orig = (y_train_enc == i).sum()
    n_sm   = (y_train_sm  == i).sum()
    nuevas = n_sm - n_orig
    print(f"          {c:15s}: {n_orig:4d} → {n_sm:4d}  (+{nuevas} sintéticas)")

print(f"\n        Tamaño original: {len(y_train_enc):5d} muestras")
print(f"        Tamaño post-SMOTE: {len(y_train_sm):5d} muestras  "
      f"(+{len(y_train_sm)-len(y_train_enc)} generadas)")

# ── 3. GRIDSEARCHCV — REGRESIÓN LOGÍSTICA ─────────────────────────────────────
"""
Misma rejilla exacta que en logistic_regression.py.
La única diferencia es que entrenamos sobre X_train_sm / y_train_sm (datos con SMOTE)
en lugar de X_train / y_train con class_weight="balanced".
Eliminamos class_weight para que el único mecanismo de balanceo sea SMOTE.
"""
print("\n[ 3/6 ] GridSearchCV — Regresión Logística con SMOTE "
      "(k=5, 12 combinaciones = 60 entrenamientos)...")

param_grid_lr = {
    "C":       [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
}

base_lr = LogisticRegression(
    solver="saga",
    class_weight=None,      # sin ponderación: SMOTE ya equilibra los datos
    max_iter=2000,
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_lr = GridSearchCV(
    estimator=base_lr,
    param_grid=param_grid_lr,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

gs_lr.fit(X_train_sm, y_train_sm)

print(f"\n        Mejor combinación: {gs_lr.best_params_}")
print(f"        Mejor Macro F1 (CV): {gs_lr.best_score_:.4f}")

mejor_lr   = gs_lr.best_estimator_
best_c_lr  = gs_lr.best_params_["C"]
best_pen   = gs_lr.best_params_["penalty"]

# ── 4. GRIDSEARCHCV — RANDOM FOREST ──────────────────────────────────────────
"""
Misma rejilla exacta que en random_forest.py.
Eliminamos class_weight="balanced" por la misma razón que en logística.
Entrenamos sobre X_train_sm / y_train_sm.
"""
print("\n[ 4/6 ] GridSearchCV — Random Forest con SMOTE "
      "(k=5, 8 combinaciones = 40 entrenamientos)...")

param_grid_rf = {
    "n_estimators": [100, 300],
    "max_features": ["sqrt", 20],
    "max_depth":    [None, 20],
}

base_rf = RandomForestClassifier(
    class_weight=None,      # sin ponderación: SMOTE ya equilibra los datos
    random_state=42,
    n_jobs=-1,
)

gs_rf = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid_rf,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

gs_rf.fit(X_train_sm, y_train_sm)

print(f"\n        Mejor combinación: {gs_rf.best_params_}")
print(f"        Mejor Macro F1 (CV): {gs_rf.best_score_:.4f}")

mejor_rf  = gs_rf.best_estimator_
best_n    = gs_rf.best_params_["n_estimators"]
best_mf   = gs_rf.best_params_["max_features"]
best_md   = gs_rf.best_params_["max_depth"]

# ── 5. EVALUACIÓN SOBRE TEST ──────────────────────────────────────────────────
"""
Evaluamos ambos modelos sobre el test original (no modificado por SMOTE).
"""
print("\n[ 5/6 ] Evaluando sobre test...")

# Regresión Logística
y_pred_lr = mejor_lr.predict(X_test)
acc_lr    = accuracy_score(y_test_enc, y_pred_lr)
f1_lr     = f1_score(y_test_enc, y_pred_lr, average="macro")
cm_lr     = confusion_matrix(y_test_enc, y_pred_lr)
report_lr = classification_report(y_test_enc, y_pred_lr,
                                   target_names=clases, output_dict=True)

# Random Forest
y_pred_rf = mejor_rf.predict(X_test)
acc_rf    = accuracy_score(y_test_enc, y_pred_rf)
f1_rf     = f1_score(y_test_enc, y_pred_rf, average="macro")
cm_rf     = confusion_matrix(y_test_enc, y_pred_rf)
report_rf = classification_report(y_test_enc, y_pred_rf,
                                   target_names=clases, output_dict=True)

print(f"\n  ── MÉTRICAS FINALES (TEST) ──────────────────────────────────")
print(f"  {'':20s}  {'Log. Reg. SMOTE':>17s}  {'Rand. Forest SMOTE':>18s}")
print(f"  {'Accuracy':20s}  {acc_lr:>17.4f}  {acc_rf:>18.4f}")
print(f"  {'Macro F1':20s}  {f1_lr:>17.4f}  {f1_rf:>18.4f}")
print(f"\n  Por clase (F1):")
for cls in clases:
    f_lr = report_lr[cls]["f1-score"]
    f_rf = report_rf[cls]["f1-score"]
    print(f"    {cls:15s}  {f_lr:>17.3f}  {f_rf:>18.3f}")

# Resultados de class_weight para la tabla comparativa
# (valores obtenidos ejecutando logistic_regression.py y random_forest.py)
F1_LR_CW  = 0.714   # Macro F1 de Regresión Logística con class_weight
F1_RF_CW  = 0.700   # Macro F1 de Random Forest con class_weight
ACC_LR_CW = 0.713
ACC_RF_CW = 0.770

F1_ABANDON_LR_CW  = report_lr["abandono"]["f1-score"]
F1_GRADUA_LR_CW   = report_lr["graduado"]["f1-score"]
F1_MATRIC_LR_CW   = report_lr["matriculado"]["f1-score"]
F1_ABANDON_RF_CW  = report_rf["abandono"]["f1-score"]
F1_GRADUA_RF_CW   = report_rf["graduado"]["f1-score"]
F1_MATRIC_RF_CW   = report_rf["matriculado"]["f1-score"]
# ──────────────────────────────────────────────────────────────────────────────

# ── 6. VISUALIZACIONES ────────────────────────────────────────────────────────
print("\n[ 6/6 ] Generando visualizaciones...")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Distribución de clases + matrices de confusión (2×2)
# ══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(16, 10))
gs1  = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.45, wspace=0.35)

# ── 6a. Distribución de clases antes y después de SMOTE ──────────────────────
ax0 = fig1.add_subplot(gs1[0, 0])

"""
Mostramos el antes/después de SMOTE para visualizar exactamente cuántas muestras
sintéticas se han generado por clase y entender el impacto sobre la distribución.
"""
counts_orig = [(y_train_enc == i).sum() for i in range(3)]
counts_sm   = [(y_train_sm  == i).sum() for i in range(3)]
x      = np.arange(3)
ancho  = 0.35

bars_orig = ax0.bar(x - ancho/2, counts_orig, ancho,
                     label="Original", color=AZUL, alpha=0.85, edgecolor="white")
bars_sm   = ax0.bar(x + ancho/2, counts_sm,   ancho,
                     label="Post-SMOTE", color=NARANJA, alpha=0.85, edgecolor="white")

for bar, val in zip(bars_orig, counts_orig):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha="center", va="bottom", fontsize=8, color=AZUL)
for bar, val_sm, val_orig in zip(bars_sm, counts_sm, counts_orig):
    nuevas = val_sm - val_orig
    label  = str(val_sm) if nuevas == 0 else f"{val_sm}\n(+{nuevas})"
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             label, ha="center", va="bottom", fontsize=8, color=NARANJA)

ax0.set_xticks(x)
ax0.set_xticklabels(clases)
ax0.set_ylabel("Número de muestras")
ax0.set_title("Distribución de clases antes y después de SMOTE\n(solo conjunto de train)",
              fontweight="bold")
ax0.legend(fontsize=9, frameon=False)
ax0.yaxis.grid(True, linestyle="--", alpha=0.4)
ax0.set_axisbelow(True)

# ── 6b. Tabla comparativa SMOTE vs class_weight ───────────────────────────────
ax1 = fig1.add_subplot(gs1[0, 1])
ax1.axis("off")

"""
Tabla resumen con los Macro F1 de ambas estrategias para los dos modelos.
Permite ver de un vistazo si SMOTE mejora o empeora respecto a class_weight.
"""
tabla_data = [
    ["",                  "Log. Reg.",               "Random Forest"],
    ["class_weight",      f"{F1_LR_CW:.3f}",         f"{F1_RF_CW:.3f}"],
    ["SMOTE",             f"{f1_lr:.3f}",             f"{f1_rf:.3f}"],
    ["Δ (SMOTE − CW)",    f"{f1_lr-F1_LR_CW:+.3f}",  f"{f1_rf-F1_RF_CW:+.3f}"],
]

table = ax1.table(
    cellText=tabla_data,
    cellLoc="center",
    loc="center",
    bbox=[0.0, 0.15, 1.0, 0.70],
)
table.auto_set_font_size(False)
table.set_fontsize(11)

# Estilo de cabecera
for j in range(3):
    cell = table[0, j]
    cell.set_facecolor("#2E75B6")
    cell.set_text_props(color="white", fontweight="bold")

# Fila de diferencias con color condicional
for j in range(1, 3):
    cell = table[3, j]
    delta_val = float(cell.get_text().get_text())
    cell.set_facecolor("#d4edda" if delta_val >= 0 else "#f8d7da")
    cell.set_text_props(fontweight="bold")

# Fila SMOTE
for j in range(3):
    table[2, j].set_facecolor("#FFF3CD")

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#CCCCCC")

ax1.set_title("Macro F1-score: SMOTE vs class_weight\n(métrica principal del proyecto)",
              fontweight="bold", pad=20)

# ── 6c. Matriz de confusión — Regresión Logística SMOTE ──────────────────────
ax2 = fig1.add_subplot(gs1[1, 0])

cm_lr_norm = cm_lr.astype(float) / cm_lr.sum(axis=1, keepdims=True)
sns.heatmap(cm_lr_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=clases, yticklabels=clases,
            ax=ax2, cbar=True, linewidths=0.5)
ax2.set_title(f"Matriz de confusión — Reg. Logística SMOTE\n"
              f"C={best_c_lr}, {best_pen.upper()}  |  Macro F1={f1_lr:.3f}",
              fontweight="bold")
ax2.set_xlabel("Predicción")
ax2.set_ylabel("Real")
for i in range(len(clases)):
    for j in range(len(clases)):
        ax2.text(j + 0.5, i + 0.75, f"n={cm_lr[i,j]}",
                 ha="center", va="center", fontsize=7, color="gray")

# ── 6d. Matriz de confusión — Random Forest SMOTE ────────────────────────────
ax3 = fig1.add_subplot(gs1[1, 1])

cm_rf_norm = cm_rf.astype(float) / cm_rf.sum(axis=1, keepdims=True)
sns.heatmap(cm_rf_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=clases, yticklabels=clases,
            ax=ax3, cbar=True, linewidths=0.5)
ax3.set_title(f"Matriz de confusión — Random Forest SMOTE\n"
              f"n={best_n}, mf={best_mf}, depth={best_md}  |  Macro F1={f1_rf:.3f}",
              fontweight="bold")
ax3.set_xlabel("Predicción")
ax3.set_ylabel("Real")
for i in range(len(clases)):
    for j in range(len(clases)):
        ax3.text(j + 0.5, i + 0.75, f"n={cm_rf[i,j]}",
                 ha="center", va="center", fontsize=7, color="gray")

fig1.suptitle("SMOTE — Distribución, comparativa y matrices de confusión",
              fontsize=13, fontweight="bold", y=1.01)

plt.savefig(f"{FIGURES_DIR}/fig_smote_comparativa.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Guardado: {FIGURES_DIR}/fig_smote_comparativa.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Métricas por clase: SMOTE vs class_weight, los dos modelos (2×2)
# ══════════════════════════════════════════════════════════════════════════════
"""
Cuatro paneles: uno por modelo × estrategia, mostrando precision/recall/F1 por clase.
Permite ver exactamente en qué clase mejora o empeora SMOTE respecto a class_weight.
"""
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
fig2.subplots_adjust(hspace=0.45, wspace=0.35)

metricas = ["precision", "recall", "f1-score"]
etiq_met = ["Precision", "Recall", "F1-score"]
x      = np.arange(len(clases))
ancho  = 0.25

configs = [
    # (ax,           report,     titulo,                                   f1_val)
    (axes[0, 0], report_lr,  f"Reg. Logística — SMOTE (C={best_c_lr}, {best_pen.upper()})",  f1_lr),
    (axes[0, 1], report_rf,  f"Random Forest — SMOTE (n={best_n}, mf={best_mf}, d={best_md})", f1_rf),
]

# Para la comparativa visual usamos los F1 por clase ya almacenados arriba.
for ax, report, titulo, f1_val in configs:
    for i, (met, label) in enumerate(zip(metricas, etiq_met)):
        vals = [report[c][met] for c in clases]
        bars = ax.bar(x + i*ancho, vals, ancho, label=label,
                      color=[AZUL, VERDE, ROJO][i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + ancho)
    ax.set_xticklabels(clases)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor")
    ax.set_title(f"{titulo}\nMacro F1 = {f1_val:.3f}", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

# ── Paneles inferiores: diferencia SMOTE − class_weight por clase y métrica ──
"""
Barras de diferencia: positivo (verde) = SMOTE mejora, negativo (rojo) = SMOTE empeora.
Permite ver de forma directa el impacto de SMOTE en cada clase y métrica.
"""
# Resultados class_weight por clase (obtenidos de logistic_regression.py y random_forest.py)
CW_LR_POR_CLASE = {
    "abandono":    {"precision": 0.81, "recall": 0.72, "f1-score": 0.764},
    "graduado":    {"precision": 0.88, "recall": 0.81, "f1-score": 0.840},
    "matriculado": {"precision": 0.46, "recall": 0.65, "f1-score": 0.537},
}
CW_RF_POR_CLASE = {
    "abandono":    {"precision": 0.816, "recall": 0.72, "f1-score": 0.765},
    "graduado":    {"precision": 0.807, "recall": 0.92, "f1-score": 0.860},
    "matriculado": {"precision": 0.541, "recall": 0.42, "f1-score": 0.473},
}
# ──────────────────────────────────────────────────────────────────────────────

for ax, report_sm, report_cw, titulo in [
    (axes[1, 0], report_lr, CW_LR_POR_CLASE, "Reg. Logística: Δ (SMOTE − class_weight) por clase"),
    (axes[1, 1], report_rf, CW_RF_POR_CLASE, "Random Forest: Δ (SMOTE − class_weight) por clase"),
]:
    for i, (met, label) in enumerate(zip(metricas, etiq_met)):
        deltas = [report_sm[c][met] - report_cw[c][met] for c in clases]
        colores = [VERDE if d >= 0 else ROJO for d in deltas]
        bars = ax.bar(x + i*ancho, deltas, ancho, label=label,
                      color=colores, alpha=0.75, edgecolor="white")
        for bar, val in zip(bars, deltas):
            if abs(val) > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.003 if val >= 0 else -0.012),
                        f"{val:+.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + ancho)
    ax.set_xticklabels(clases)
    ax.set_ylabel("Diferencia (SMOTE − class_weight)")
    ax.set_title(titulo, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.annotate("verde = SMOTE mejor  |  rojo = class_weight mejor",
                xy=(0.5, -0.13), xycoords="axes fraction",
                ha="center", fontsize=8, color="gray", style="italic")

fig2.suptitle("Métricas por clase — SMOTE vs class_weight (Reg. Logística y Random Forest)",
              fontsize=13, fontweight="bold", y=1.01)

plt.savefig(f"{FIGURES_DIR}/fig_smote_metricas_clase.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Guardado: {FIGURES_DIR}/fig_smote_metricas_clase.png")

print("\n[ DONE ] Análisis SMOTE completado.")
print(f"\n  Resumen final:")
print(f"    Reg. Logística  — class_weight Macro F1: {F1_LR_CW:.3f} | SMOTE Macro F1: {f1_lr:.3f}  "
      f"(Δ={f1_lr-F1_LR_CW:+.3f})")
print(f"    Random Forest   — class_weight Macro F1: {F1_RF_CW:.3f} | SMOTE Macro F1: {f1_rf:.3f}  "
      f"(Δ={f1_rf-F1_RF_CW:+.3f})")
