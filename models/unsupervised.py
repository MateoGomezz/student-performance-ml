"""
unsupervised.py — Tarea 3: Aprendizaje no supervisado
=======================================================
Pipeline completo:
    1. Selección de variables numéricas con significado real
    2. PCA sobre las 14 variables → scree plot → elegir N componentes (80% varianza)
    3. Visualización 2D (PC1 vs PC2) coloreada por etiqueta real
    4. Análisis de loadings de PC1 y PC2
    5. Elección de K óptimo: codo + Silhouette
    6. K-Means sobre los N componentes PCA (Opción B)
    7. Visualización clusters en 2D vs etiquetas reales (side by side)
    8. Heatmap de perfiles de clusters (valores originales)
    9. Estabilidad ARI

DECISIÓN CLAVE — dónde aplica K-Means:
    K-Means se aplica sobre los N primeros componentes PCA que acumulan el 80%
    de varianza explicada (espacio reducido y sin correlaciones). La visualización
    se hace siempre en PC1 vs PC2 (proyección 2D), con nota explícita de que el
    clustering ocurrió en el espacio de N componentes.

Requiere en data_processed/:
    X_train.csv   → features completas preprocesadas (estandarizadas)
    y_train.csv   → columna 'objetivo' para colorear

Salidas en figures/:
    01_scree_plot.png
    02_pca_scatter_target.png
    03_pca_loadings.png
    04_elbow_silhouette.png
    05_clusters_vs_target.png
    06_cluster_profiles_heatmap.png
    07_stability_ari.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from itertools import combinations

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processed")
FIGURES_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
K_RANGE       = range(2, 9)
N_STABILITY   = 30
VAR_THRESHOLD = 0.80   # umbral de varianza acumulada para elegir N componentes

os.makedirs(FIGURES_DIR, exist_ok=True)

# Paletas consistentes en todo el script
COLOR_TARGET   = {"abandono": "#E05252", "graduado": "#4CAF82", "matriculado": "#5B9BD5"}
COLOR_CLUSTERS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                  "#0072B2", "#D55E00", "#CC79A7", "#999999"]

plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.2,
    "font.family":       "sans-serif",
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
})


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 1 — CARGA Y SELECCIÓN DE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PASO 1 — Carga y selección de variables")
print("=" * 65)

X_full  = pd.read_csv(f"{DATA_DIR}/X_train.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")
target  = y_train["objetivo"].str.lower().str.strip().values

# Subconjunto razonado:
#   - Variables numéricas continuas de rendimiento académico (1er y 2do semestre)
#   - Variables de perfil de acceso (notas previas, edad)
#   - Variables binarias socioeconómicas con semántica propia (solo 3: efecto controlado)
#
# Se excluyen:
#   - Columnas OHE (~70 cols): su efecto acumulado distorsiona distancias euclídeas
#   - Variables macroeconómicas: no describen al estudiante individualmente
#   - asignaturas_sin_evaluacion / convalidadas: distribución concentrada en 0

FEATURES = [
    # Rendimiento 1er semestre
    "asignaturas_1sem_matriculadas",
    "asignaturas_1sem_evaluadas",
    "asignaturas_1sem_aprobadas",
    "nota_media_1sem",
    # Rendimiento 2do semestre (sin riesgo de leakage en no supervisado)
    "asignaturas_2sem_matriculadas",
    "asignaturas_2sem_evaluadas",
    "asignaturas_2sem_aprobadas",
    "nota_media_2sem",
    # Perfil de acceso
    "nota_admision",
    "nota_cualificacion_previa",
    "edad_al_matricularse",
    # Situación económica (binarias con semántica propia, efecto en distancias controlado)
    "becado",
    "matricula_al_dia",
    "deudor",
]

# Robustez ante posibles diferencias de nombre en el CSV
FEATURES = [f for f in FEATURES if f in X_full.columns]
X = X_full[FEATURES].copy()

print(f"  Variables seleccionadas : {len(FEATURES)}")
print(f"  Observaciones           : {X.shape[0]}")
print(f"  Features                : {FEATURES}")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2 — PCA COMPLETO + SCREE PLOT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 2 — PCA completo + Scree Plot")
print("=" * 65)

pca_full   = PCA(random_state=RANDOM_STATE)
pca_full.fit(X)
explained  = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

N_COMP = int(np.argmax(cumulative >= VAR_THRESHOLD)) + 1
print(f"  Componentes para {VAR_THRESHOLD*100:.0f}% varianza : {N_COMP}")
print(f"  Varianza acumulada con {N_COMP} comp     : {cumulative[N_COMP-1]*100:.1f}%")
print(f"  Varianza PC1 : {explained[0]*100:.1f}%  |  PC2 : {explained[1]*100:.1f}%")

# ── Figura 01: Scree plot ────────────────────────────────────────────────────
n_show = len(FEATURES)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bar_colors = ["#E05252" if i < N_COMP else "#5B9BD5" for i in range(n_show)]
axes[0].bar(range(1, n_show + 1), explained[:n_show] * 100,
            color=bar_colors, alpha=0.85, edgecolor="white", zorder=3)
axes[0].set_xlabel("Componente principal")
axes[0].set_ylabel("Varianza explicada (%)")
axes[0].set_title("Varianza explicada por componente\n"
                  "(rojo = retenidos, azul = descartados)")
axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[1].plot(range(1, n_show + 1), cumulative[:n_show] * 100,
             marker="o", color="#E05252", linewidth=2, markersize=6, zorder=3)
axes[1].axhline(VAR_THRESHOLD * 100, color="#888888", linestyle="--", linewidth=1.3,
                label=f"Umbral {VAR_THRESHOLD*100:.0f}%")
axes[1].axvline(N_COMP, color="#4CAF82", linestyle="--", linewidth=1.5,
                label=f"N={N_COMP} componentes ({cumulative[N_COMP-1]*100:.1f}%)")
axes[1].fill_between(range(1, N_COMP + 1), cumulative[:N_COMP] * 100,
                     alpha=0.12, color="#4CAF82")
axes[1].set_xlabel("Número de componentes")
axes[1].set_ylabel("Varianza acumulada (%)")
axes[1].set_title("Varianza acumulada — elección de N")
axes[1].legend()
axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle("PCA — Scree Plot", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/01_scree_plot.png", bbox_inches="tight")
plt.close()
print(f"  → Guardado: 01_scree_plot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 3 — VISUALIZACIÓN 2D COLOREADA POR ETIQUETA REAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 3 — Scatter 2D coloreado por etiqueta real")
print("=" * 65)

# PCA 2D — solo para visualización
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d   = pca_2d.fit_transform(X)
var_pc1 = pca_2d.explained_variance_ratio_[0] * 100
var_pc2 = pca_2d.explained_variance_ratio_[1] * 100

# PCA N componentes — para clustering (Opción B)
pca_Nd = PCA(n_components=N_COMP, random_state=RANDOM_STATE)
X_Nd   = pca_Nd.fit_transform(X)

fig, ax = plt.subplots(figsize=(9, 6.5))
for clase, color in COLOR_TARGET.items():
    mask = target == clase
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, label=clase.capitalize(),
               alpha=0.5, s=16, edgecolors="none")

ax.set_xlabel(f"PC1 ({var_pc1:.1f}% varianza explicada)")
ax.set_ylabel(f"PC2 ({var_pc2:.1f}% varianza explicada)")
ax.set_title("Proyección PCA 2D — coloreado por situación académica real")
ax.legend(title="Situación", markerscale=2.5)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/02_pca_scatter_target.png", bbox_inches="tight")
plt.close()
print(f"  → Guardado: 02_pca_scatter_target.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 4 — ANÁLISIS DE LOADINGS PC1 y PC2
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 4 — Análisis de loadings")
print("=" * 65)

loadings_df = pd.DataFrame(
    pca_2d.components_.T,
    index=FEATURES,
    columns=["PC1", "PC2"]
)

print("\n  Loadings PC1 (ordenados por peso):")
print(loadings_df["PC1"].sort_values(ascending=False).round(3).to_string())
print("\n  Loadings PC2 (ordenados por peso):")
print(loadings_df["PC2"].sort_values(ascending=False).round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for i, pc in enumerate(["PC1", "PC2"]):
    data   = loadings_df[pc].sort_values()
    colors = ["#E05252" if v < 0 else "#4CAF82" for v in data.values]
    axes[i].barh(data.index, data.values, color=colors,
                 edgecolor="white", height=0.65)
    axes[i].axvline(0, color="#444444", linewidth=0.9)
    var_pct = pca_2d.explained_variance_ratio_[i] * 100
    axes[i].set_title(f"Loadings {pc}  ({var_pct:.1f}% varianza)")
    axes[i].set_xlabel("Peso del loading")
    axes[i].set_xlim(-0.70, 0.70)
    for val, name in zip(data.values, data.index):
        offset = 0.02 if val >= 0 else -0.02
        ha     = "left" if val >= 0 else "right"
        axes[i].text(val + offset, name, f"{val:.2f}",
                     va="center", ha=ha, fontsize=7.5, color="#333333")

plt.suptitle("Contribución de cada variable a PC1 y PC2\n"
             "(verde = contribución positiva  |  rojo = negativa)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/03_pca_loadings.png", bbox_inches="tight")
plt.close()
print(f"  → Guardado: 03_pca_loadings.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 5 — ELECCIÓN DE K: CODO + SILHOUETTE  (K-Means sobre X_Nd)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 5 — Elección de K óptimo")
print("=" * 65)

inertias    = []
silhouettes = []
K_list      = list(K_RANGE)

for k in K_list:
    km  = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    lbl = km.fit_predict(X_Nd)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_Nd, lbl)
    silhouettes.append(sil)
    print(f"  K={k}  inercia={km.inertia_:8.1f}  silhouette={sil:.4f}")

best_k = K_list[int(np.argmax(silhouettes))]
print(f"\n  → K óptimo (Silhouette máximo): {best_k}  (score={max(silhouettes):.4f})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(K_list, inertias, marker="o", color="#5B9BD5",
             linewidth=2, markersize=7, zorder=3)
axes[0].axvline(best_k, color="#E05252", linestyle="--", linewidth=1.4,
                label=f"K óptimo = {best_k}")
axes[0].set_xlabel("Número de clusters K")
axes[0].set_ylabel("Inercia (Within-Cluster SS)")
axes[0].set_title("Método del codo")
axes[0].legend()
axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[1].plot(K_list, silhouettes, marker="s", color="#E05252",
             linewidth=2, markersize=7, zorder=3)
axes[1].axvline(best_k, color="#4CAF82", linestyle="--", linewidth=1.4,
                label=f"K óptimo = {best_k}  (score={max(silhouettes):.4f})")
axes[1].set_xlabel("Número de clusters K")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score por K")
axes[1].legend()
axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle(f"Selección del número óptimo de clusters\n"
             f"(K-Means evaluado sobre {N_COMP} componentes PCA)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/04_elbow_silhouette.png", bbox_inches="tight")
plt.close()
print(f"  → Guardado: 04_elbow_silhouette.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 6 — K-MEANS FINAL sobre N componentes PCA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"PASO 6 — K-Means final  K={best_k}  sobre {N_COMP} componentes PCA")
print("=" * 65)

km_final       = KMeans(n_clusters=best_k, init="k-means++",
                        n_init=10, random_state=RANDOM_STATE)
cluster_labels = km_final.fit_predict(X_Nd)

sizes = pd.Series(cluster_labels).value_counts().sort_index()
print("  Tamaño de clusters:")
for k, n in sizes.items():
    print(f"    Cluster {k}: {n} estudiantes ({n/len(cluster_labels)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 7 — CLUSTERS vs ETIQUETAS REALES (side by side)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 7 — Clusters vs etiquetas reales (side by side)")
print("=" * 65)

# Proyectar centroides del espacio N-D al espacio 2D para visualizarlos
centroids_Nd   = km_final.cluster_centers_              # (best_k, N_COMP)
centroids_orig = pca_Nd.inverse_transform(centroids_Nd) # (best_k, 14) reconstruido
centroids_2d   = pca_2d.transform(centroids_orig)       # (best_k, 2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# ── Panel izquierdo: clusters K-Means ───────────────────────────────────────
for k in range(best_k):
    mask = cluster_labels == k
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=[COLOR_CLUSTERS[k % len(COLOR_CLUSTERS)]],
                    label=f"Cluster {k}  (n={mask.sum()})",
                    alpha=0.55, s=16, edgecolors="none")
axes[0].scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                c="black", marker="X", s=140, zorder=6,
                linewidths=0.8, label="Centroides (proy.)")
axes[0].set_xlabel(f"PC1 ({var_pc1:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var_pc2:.1f}%)")
axes[0].set_title(f"K-Means  K={best_k}\n"
                  f"(clustering en {N_COMP} componentes, visual en 2D)")
axes[0].legend(fontsize=8, markerscale=2, loc="best")

# ── Panel derecho: etiquetas reales ─────────────────────────────────────────
for clase, color in COLOR_TARGET.items():
    mask = target == clase
    axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=color, label=clase.capitalize(),
                    alpha=0.5, s=16, edgecolors="none")
axes[1].set_xlabel(f"PC1 ({var_pc1:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_pc2:.1f}%)")
axes[1].set_title("Situación académica real\n(Abandono / Matriculado / Graduado)")
axes[1].legend(fontsize=9, markerscale=2.5, loc="best")

fig.text(0.5, -0.03,
         f"Nota metodológica: K-Means se aplicó sobre los {N_COMP} primeros componentes PCA "
         f"({cumulative[N_COMP-1]*100:.1f}% de varianza explicada). "
         "La proyección en 2D es únicamente para visualización.",
         ha="center", fontsize=8.5, color="#555555", style="italic")

plt.suptitle("Clusters obtenidos (K-Means)  vs  situación académica real",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/05_clusters_vs_target.png", bbox_inches="tight")
plt.close()
print(f"  → Guardado: 05_clusters_vs_target.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 8 — HEATMAP PERFILES + COMPOSICIÓN POR ETIQUETA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PASO 8 — Heatmap de perfiles de clusters")
print("=" * 65)

X_interp             = X.copy()
X_interp["cluster"]  = cluster_labels
X_interp["objetivo"] = target

global_mean   = X[FEATURES].mean()
cluster_means = X_interp.groupby("cluster")[FEATURES].mean()
delta         = cluster_means.subtract(global_mean)

print("\n  Desviación respecto a la media global (unidades estandarizadas):")
print(delta.round(3).to_string())

rename_map = {
    "asignaturas_1sem_matriculadas": "1sem_matr",
    "asignaturas_1sem_evaluadas":    "1sem_eval",
    "asignaturas_1sem_aprobadas":    "1sem_apr",
    "nota_media_1sem":               "nota_1sem",
    "asignaturas_2sem_matriculadas": "2sem_matr",
    "asignaturas_2sem_evaluadas":    "2sem_eval",
    "asignaturas_2sem_aprobadas":    "2sem_apr",
    "nota_media_2sem":               "nota_2sem",
    "nota_admision":                 "nota_adm",
    "nota_cualificacion_previa":     "nota_prev",
    "edad_al_matricularse":          "edad",
    "becado":                        "becado",
    "matricula_al_dia":              "mat_dia",
    "deudor":                        "deudor",
}
delta_plot = delta.rename(columns=rename_map)
cross = pd.crosstab(X_interp["cluster"], X_interp["objetivo"],
                    normalize="index") * 100

fig = plt.figure(figsize=(17, 4 + best_k * 0.7))
gs  = gridspec.GridSpec(1, 2, width_ratios=[3.2, 1], wspace=0.06)
ax_heat  = fig.add_subplot(gs[0])
ax_cross = fig.add_subplot(gs[1])

sns.heatmap(
    delta_plot, annot=True, fmt="+.2f",
    cmap="RdYlGn", center=0, vmin=-2, vmax=2,
    linewidths=0.4, linecolor="#CCCCCC", ax=ax_heat,
    cbar_kws={"label": "Desviación vs media global (std)", "shrink": 0.8}
)
ax_heat.set_title("Perfil de cada cluster — desviación respecto a la media global",
                  fontweight="bold", pad=10)
ax_heat.set_xlabel("Variable")
ax_heat.set_ylabel("Cluster")
ax_heat.set_yticklabels([f"Cluster {k}" for k in range(best_k)], rotation=0)
ax_heat.tick_params(axis="x", rotation=40)

sns.heatmap(
    cross, annot=True, fmt=".1f",
    cmap="Blues", linewidths=0.4, linecolor="#CCCCCC",
    ax=ax_cross,
    cbar_kws={"label": "% estudiantes", "shrink": 0.8}
)
ax_cross.set_title("Composición\npor situación (%)", fontweight="bold", pad=10)
ax_cross.set_xlabel("Situación")
ax_cross.set_ylabel("")
ax_cross.set_yticklabels([f"Cluster {k}" for k in range(best_k)], rotation=0)
ax_cross.tick_params(axis="x", rotation=30)

plt.savefig(f"{FIGURES_DIR}/06_cluster_profiles_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  → Guardado: 06_cluster_profiles_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 9 — ESTABILIDAD ARI
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"PASO 9 — Estabilidad  ({N_STABILITY} semillas, K={best_k})")
print("=" * 65)

all_labels = []
for seed in range(N_STABILITY):
    km_tmp = KMeans(n_clusters=best_k, init="k-means++",
                    n_init=1, random_state=seed)
    all_labels.append(km_tmp.fit_predict(X_Nd))

ari_scores = [
    adjusted_rand_score(li, lj)
    for (_, li), (_, lj) in combinations(enumerate(all_labels), 2)
]

ari_mean = np.mean(ari_scores)
ari_std  = np.std(ari_scores)
ari_min  = np.min(ari_scores)

print(f"  ARI medio  : {ari_mean:.4f}")
print(f"  ARI std    : {ari_std:.4f}")
print(f"  ARI mínimo : {ari_min:.4f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(ari_scores, bins=25, color="#5B9BD5", edgecolor="white", alpha=0.85)
ax.axvline(ari_mean, color="#E05252", linewidth=2, linestyle="--",
           label=f"Media = {ari_mean:.3f}")
ax.axvline(ari_mean - ari_std, color="#888888", linewidth=1.2, linestyle=":",
           label=f"±1σ  (σ = {ari_std:.3f})")
ax.axvline(ari_mean + ari_std, color="#888888", linewidth=1.2, linestyle=":")
ax.set_xlabel("Adjusted Rand Index (ARI)")
ax.set_ylabel("Frecuencia")
ax.set_title(f"Estabilidad de los clusters  K={best_k} — "
             f"{N_STABILITY} inicializaciones aleatorias", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/07_stability_ari.png", bbox_inches="tight")
plt.close()
print(f"  → Guardado: 07_stability_ari.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("RESUMEN FINAL")
print("=" * 65)
print(f"  Variables seleccionadas          : {len(FEATURES)}")
print(f"  Componentes PCA retenidos (≥80%) : {N_COMP}  "
      f"({cumulative[N_COMP-1]*100:.1f}% varianza)")
print(f"  Varianza capturada PC1+PC2       : {(var_pc1+var_pc2):.1f}%  (solo visual)")
print(f"  K óptimo (Silhouette)            : {best_k}  "
      f"(score={max(silhouettes):.4f})")
print(f"  Estabilidad ARI                  : {ari_mean:.4f} ± {ari_std:.4f}  "
      f"(mín={ari_min:.4f})")
print(f"  Figuras guardadas en             : ./{FIGURES_DIR}/")
print("=" * 65)
