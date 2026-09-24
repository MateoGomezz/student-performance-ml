"""
Clasificación multiclase con Regresión Logística.

Flujo:
    1. Cargar X_train, X_test, y_train, y_test
    2. GridSearchCV (k=5 folds) → 6 valores de C × 2 regularizaciones = 60 entrenamientos
    3. Reentrenar el mejor modelo sobre todo el train
    4. Evaluar sobre test → métricas finales
    5. Visualizar: matriz de confusión + top features por clase
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Estilo global ────────────────────────────────────────────────────────────
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

# ── 1. CARGA DE DATOS ────────────────────────────────────────────────────────
"""
Leemos los 4 CSVs que generamos en preprocessing.py. Usamos la columna
objetivo_encoded ya que sklearn necesita números, no texto.

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

# 2. GRIDSEARCHCV : obtener la mejor regularización y mejor valor de c
"""
Hacemos cross-validation con distintos valores de C y probando la penalización
de Ridge y la de Lasso. Usamos el solver saga que es el único en sklearn que
soporta L1 multiclase al mismo tiempo. Usamos también class_weight para balancear
las clases.
Obtenemos finalmente la combinación (c,l?) que mejores resultados ha tenido
en cross-validation
"""

print("\n[ 2/5 ] GridSearchCV (k=5, 12 combinaciones = 60 entrenamientos)...")

param_grid = {
    "C":       [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
}

base_model = LogisticRegression(
    solver="saga",          # único solver que soporta l1 + multinomial
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",     # métrica principal: macro F1
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train_enc)

print(f"\n        Mejor combinación: {grid_search.best_params_}")
print(f"        Mejor macro F1 (CV): {grid_search.best_score_:.4f}")

# ── 3. REENTRENAR SOBRE TODO EL TRAIN ────────────────────────────────────────
"""
Reentrenamos todo el conjunto de train con esos parámetros óptimos encontrados
"""
print("\n[ 3/5 ] Reentrenando mejor modelo sobre todo el train...")
mejor_modelo = grid_search.best_estimator_
# GridSearchCV ya refit=True por defecto, así que mejor_modelo ya está reentrenado

# ── 4. EVALUACIÓN SOBRE TEST ─────────────────────────────────────────────────
"""
Terminamos evaluando sobre test y obteniendo distintas métricas para ver
el comportamiento obtenido. Con esas métricas podremos hacer varias
visualizaciones finales.
"""
print("\n[ 4/5 ] Evaluando sobre test...")
y_pred = mejor_modelo.predict(X_test)

acc     = accuracy_score(y_test_enc, y_pred)
f1_macro = f1_score(y_test_enc, y_pred, average="macro")
cm      = confusion_matrix(y_test_enc, y_pred)
report  = classification_report(y_test_enc, y_pred,
                                 target_names=clases, output_dict=True)

print(f"\n  ── MÉTRICAS FINALES (TEST) ──────────────────────────")
print(f"  Accuracy:       {acc:.4f}")
print(f"  Macro F1-score: {f1_macro:.4f}  ← métrica principal")
print(f"\n  Por clase:")
for cls in clases:
    r = report[cls]
    print(f"    {cls:15s}  precision={r['precision']:.3f}  "
          f"recall={r['recall']:.3f}  F1={r['f1-score']:.3f}  "
          f"n={r['support']:.0f}")


"""
Tenemos los códigos de las visualizaciones que vamos ha explicar en el notebook
de resumen de este modelo en la carpeta resumenes. Nos permitirán ver más claramente
cómo se han obtenido los hiperparámetros óptimos para regularizar y varias métricas
sobre el rendimiento de nuestro modelo en el conjunto de test para sacar conclusiones
finales.
"""
# ── 5. VISUALIZACIONES ───────────────────────────────────────────────────────
print("\n[ 5/5 ] Generando visualizaciones...")

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 5a. Resultados CV: F1 medio por combinación ──────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_l1 = cv_results[cv_results["param_penalty"] == "l1"].sort_values("param_C")
cv_l2 = cv_results[cv_results["param_penalty"] == "l2"].sort_values("param_C")
c_vals = [0.001, 0.01, 0.1, 1, 10, 100]

ax0.plot(range(len(c_vals)), cv_l1["mean_test_score"].values,
         marker="o", color=AZUL, label="L1 (Lasso)", linewidth=2)
ax0.fill_between(range(len(c_vals)),
                 cv_l1["mean_test_score"] - cv_l1["std_test_score"],
                 cv_l1["mean_test_score"] + cv_l1["std_test_score"],
                 alpha=0.15, color=AZUL)
ax0.plot(range(len(c_vals)), cv_l2["mean_test_score"].values,
         marker="s", color=ROJO, label="L2 (Ridge)", linewidth=2)
ax0.fill_between(range(len(c_vals)),
                 cv_l2["mean_test_score"] - cv_l2["std_test_score"],
                 cv_l2["mean_test_score"] + cv_l2["std_test_score"],
                 alpha=0.15, color=ROJO)

ax0.set_xticks(range(len(c_vals)))
ax0.set_xticklabels([str(c) for c in c_vals])
ax0.set_xlabel("Valor de C (inverso de regularización)")
ax0.set_ylabel("Macro F1-score (CV)")
ax0.set_title("Búsqueda de hiperparámetros — GridSearchCV", fontweight="bold")
ax0.legend(fontsize=9, frameon=False)
ax0.yaxis.grid(True, linestyle="--", alpha=0.4)
ax0.set_axisbelow(True)

# Marcar el mejor
best_c    = grid_search.best_params_["C"]
best_pen  = grid_search.best_params_["penalty"]
best_idx  = c_vals.index(best_c)
best_score = grid_search.best_score_
ax0.axvline(best_idx, color="gray", linestyle=":", linewidth=1.2)
ax0.annotate(f"Mejor: C={best_c}\n{best_pen.upper()}\nF1={best_score:.3f}",
             xy=(best_idx, best_score),
             xytext=(best_idx + 0.3, best_score - 0.03),
             fontsize=8, color="gray",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# ── 5b. Matriz de confusión ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=clases, yticklabels=clases,
            ax=ax1, cbar=True, linewidths=0.5)
ax1.set_title("Matriz de confusión (normalizada por fila)", fontweight="bold")
ax1.set_xlabel("Predicción")
ax1.set_ylabel("Real")

# Añadir conteos absolutos en cada celda
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
ax2.legend(fontsize=9, frameon=False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

# ── 5d. Top features por clase (coeficientes) ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

feature_names = X_train.columns.tolist()
coefs = mejor_modelo.coef_   # shape: (n_clases, n_features)
top_n = 10

# Nos quedamos con la clase más difícil: matriculado (índice 2)
idx_mat = 2
coef_mat = coefs[idx_mat]
top_idx  = np.argsort(np.abs(coef_mat))[-top_n:]
top_names = [feature_names[i] for i in top_idx]
top_vals  = coef_mat[top_idx]

colores_barras = [ROJO if v > 0 else AZUL for v in top_vals]
ax3.barh(range(top_n), top_vals, color=colores_barras, edgecolor="white", alpha=0.85)
ax3.set_yticks(range(top_n))
ax3.set_yticklabels(top_names, fontsize=8)
ax3.axvline(0, color="gray", linewidth=0.8)
ax3.set_title(f"Top {top_n} features — clase 'matriculado'\n"
              f"(azul = empuja a NO matriculado, rojo = empuja a matriculado)",
              fontweight="bold", fontsize=10)
ax3.set_xlabel("Coeficiente")
ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)

# ── Título general ────────────────────────────────────────────────────────────
fig.suptitle(
    f"Regresión Logística — Mejor modelo: C={best_c}, {best_pen.upper()}  |  "
    f"Accuracy={acc:.3f}  |  Macro F1={f1_macro:.3f}",
    fontsize=12, fontweight="bold", y=1.01
)

plt.savefig(f"{FIGURES_DIR}/fig_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Guardado: {FIGURES_DIR}/fig_logistic_regression.png")
print("\n[ DONE ] Regresión Logística completada.")

def plot_colinealidad(X_train, mejor_modelo, feature_names):
    """
    Genera una figura de dos paneles para analizar la colinealidad del dataset
    y su relación con la preferencia de Lasso sobre Ridge.

    Parámetros
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento (ya estandarizadas).
    mejor_modelo : LogisticRegression
        Modelo óptimo entrenado (Lasso C=0.1). Se usa para extraer los pesos.
    feature_names : list[str]
        Lista de nombres de columnas en el mismo orden que X_train.
    """

    # ── Features continuas de interés ─────────────────────────────────────────
    # Seleccionamos las variables académicas y contextuales numéricas puras.
    # Excluimos las dummies OHE (binarias 0/1) porque su correlación de Pearson
    # no es interpretable de la misma forma que entre variables continuas.
    FEATURES_CONTINUAS = [
        "asignaturas_1sem_matriculadas", "asignaturas_1sem_evaluadas",
        "asignaturas_1sem_aprobadas",    "asignaturas_1sem_sin_evaluacion",
        "nota_media_1sem",               "asignaturas_1sem_convalidadas",
        "asignaturas_2sem_matriculadas", "asignaturas_2sem_evaluadas",
        "asignaturas_2sem_aprobadas",    "asignaturas_2sem_sin_evaluacion",
        "nota_media_2sem",               "asignaturas_2sem_convalidadas",
        "nota_cualificacion_previa",     "nota_admision",
        "edad_al_matricularse",          "tasa_desempleo",
        "tasa_inflacion",                "pib",
    ]
    # Filtrar solo las que realmente estén en X_train (por si el preprocesado cambia)
    features_plot = [f for f in FEATURES_CONTINUAS if f in X_train.columns]

    # ── Etiquetas cortas para que quepan en el eje ────────────────────────────
    etiquetas_cortas = {
        "asignaturas_1sem_matriculadas":  "1s_matr",
        "asignaturas_1sem_evaluadas":     "1s_eval",
        "asignaturas_1sem_aprobadas":     "1s_aprob",
        "asignaturas_1sem_sin_evaluacion":"1s_sin_eval",
        "nota_media_1sem":                "nota_1s",
        "asignaturas_1sem_convalidadas":  "1s_conv",
        "asignaturas_2sem_matriculadas":  "2s_matr",
        "asignaturas_2sem_evaluadas":     "2s_eval",
        "asignaturas_2sem_aprobadas":     "2s_aprob",
        "asignaturas_2sem_sin_evaluacion":"2s_sin_eval",
        "nota_media_2sem":                "nota_2s",
        "asignaturas_2sem_convalidadas":  "2s_conv",
        "nota_cualificacion_previa":      "nota_prev",
        "nota_admision":                  "nota_adm",
        "edad_al_matricularse":           "edad",
        "tasa_desempleo":                 "desempleo",
        "tasa_inflacion":                 "inflacion",
        "pib":                            "pib",
    }

    # ── Grupos semánticos para el panel derecho ────────────────────────────────
    # Agrupamos todas las features del modelo en bloques semánticos para ver
    # cuántas ha eliminado Lasso en cada bloque.
    grupos = {
        "Académicas 1er sem":  [f for f in feature_names if "1sem" in f],
        "Académicas 2º sem":   [f for f in feature_names if "2sem" in f],
        "Admisión / notas":    ["nota_cualificacion_previa", "nota_admision",
                                "orden_solicitud"],
        "Sociodemográficas":   ["edad_al_matricularse", "genero", "internacional",
                                "desplazado", "becado",
                                "necesidades_educativas_especiales",
                                "deudor", "matricula_al_dia",
                                "asistencia_diurna_vespertina"],
        "Familia / origen":    [f for f in feature_names
                                if any(x in f for x in
                                       ["madre", "padre", "nacionalidad"])],
        "Curso / solicitud":   [f for f in feature_names
                                if any(x in f for x in
                                       ["curso", "modo_solicitud",
                                        "estado_civil", "cualificacion_previa_"])],
        "Macroeconómicas":     ["tasa_desempleo", "tasa_inflacion", "pib"],
    }

    # ── Calcular pesos nulos por grupo (promedio sobre las 3 clases) ───────────
    # Un peso se considera nulo si |w| < 1e-4 (umbral numérico estándar para Lasso)
    coefs = mejor_modelo.coef_          # shape: (3, n_features)
    umbral_nulo = 1e-4

    nombres_grupos, n_total, n_nulos = [], [], []
    for nombre, feats in grupos.items():
        idxs = [feature_names.index(f) for f in feats if f in feature_names]
        if not idxs:
            continue
        pesos_grupo = coefs[:, idxs]    # (3 clases, n_feats_grupo)
        total  = len(idxs) * 3          # 3 clases × n_features del grupo
        nulos  = int(np.sum(np.abs(pesos_grupo) < umbral_nulo))
        nombres_grupos.append(nombre)
        n_total.append(total)
        n_nulos.append(nulos)

    n_activos = [t - n for t, n in zip(n_total, n_nulos)]
    pct_nulos = [100 * n / t for n, t in zip(n_nulos, n_total)]

    # ── Figura ─────────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={"width_ratios": [1.1, 1]}
    )

    # ── Panel izquierdo: heatmap de correlaciones ──────────────────────────────
    corr = X_train[features_plot].corr()
    labels = [etiquetas_cortas.get(f, f) for f in features_plot]

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True   # solo triángulo inferior

    sns.heatmap(
        corr,
        mask=mask,
        ax=ax_left,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        annot=True, fmt=".1f",
        annot_kws={"size": 7},
        linewidths=0.3,
        square=True,
        cbar_kws={"shrink": 0.7, "label": "Correlación de Pearson"},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax_left.set_title(
        "Correlaciones entre features continuas clave\n"
        "(bloques oscuros = grupos de variables redundantes)",
        fontweight="bold", fontsize=11, pad=12
    )
    ax_left.tick_params(axis="x", rotation=45, labelsize=8)
    ax_left.tick_params(axis="y", rotation=0,  labelsize=8)

    # Recuadros para destacar los dos bloques de semestres
    # Bloque 1er semestre: filas/columnas 0–5 (6 features)
    for ax_rect, xy, wh in [
        (ax_left, (0, 0),   (6, 6)),    # 1er semestre
        (ax_left, (6, 6),   (6, 6)),    # 2º semestre
    ]:
        rect = plt.Rectangle(
            xy, wh[0], wh[1],
            fill=False, edgecolor="#FF9800", linewidth=2.5, linestyle="--"
        )
        ax_rect.add_patch(rect)

    ax_left.text(3,   -0.6, "1er sem",  ha="center", color="#FF9800",
                 fontsize=8, fontweight="bold")
    ax_left.text(9,   -0.6, "2º sem",   ha="center", color="#FF9800",
                 fontsize=8, fontweight="bold")

    # ── Panel derecho: pesos nulos por grupo ───────────────────────────────────
    y_pos = np.arange(len(nombres_grupos))
    bar_h = 0.55

    bars_act  = ax_right.barh(y_pos,  n_activos, bar_h,
                               color=AZUL,  alpha=0.85, label="Pesos activos  |w| ≥ 1e-4")
    bars_nulo = ax_right.barh(y_pos,  n_nulos,   bar_h,
                               left=n_activos,
                               color="#BDBDBD", alpha=0.85, label="Pesos ≈ 0  (Lasso eliminó)")

    # Porcentaje de eliminación al final de cada barra
    for i, (pct, total) in enumerate(zip(pct_nulos, n_total)):
        ax_right.text(
            total + 0.5, y_pos[i],
            f"{pct:.0f}% nulos",
            va="center", fontsize=9,
            color="#555555" if pct < 50 else ROJO,
            fontweight="bold" if pct >= 50 else "normal"
        )

    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(nombres_grupos, fontsize=10)
    ax_right.set_xlabel("Número de pesos  (3 clases × n_features del grupo)", fontsize=10)
    ax_right.set_title(
        "Pesos eliminados por Lasso (C=0.1) por grupo semántico\n"
        "(un % alto de nulos indica redundancia en ese grupo)",
        fontweight="bold", fontsize=11, pad=12
    )
    ax_right.legend(fontsize=9, frameon=False, loc="lower right")
    ax_right.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax_right.set_axisbelow(True)
    # Quitar spine derecho/superior (ya desactivados globalmente, pero por si acaso)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.suptitle(
        "Análisis de colinealidad — Justificación de Lasso sobre Ridge",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_colinealidad.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Guardado: {FIGURES_DIR}/fig_colinealidad.png")


# ── Llamada a la función ──────────────────────────────────────────────────────
print("\n[ 6/5 ] Analizando colinealidad...")
plot_colinealidad(X_train, mejor_modelo, feature_names=X_train.columns.tolist())
