"""
Regresión lineal para predicción de nota_media_2sem.

Flujo:
    1. Cargar X_train_reg, X_test_reg, y_train, y_test
    2. GridSearchCV (k=5 folds) → 6 valores de alpha × 2 regularizaciones = 60 entrenamientos
    3. Reentrenar Ridge óptimo y Lasso óptimo sobre todo el train
    4. Evaluar sobre test → métricas finales (RMSE y R²)
    5. Visualizar: curva alpha vs RMSE (CV), predicciones vs reales por clase,
                   análisis de residuos, top coeficientes Lasso
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Estilo global ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
})
AZUL  = "#2E75B6"
ROJO  = "#E8593C"
VERDE = "#2E8B57"
GRIS  = "#7F7F7F"

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processed")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. CARGA DE DATOS ─────────────────────────────────────────────────────────
"""
Usamos los ficheros preparados expresamente para regresión en preprocessing.py.
A diferencia de clasificación, cargamos X_train_reg y X_test_reg, que NO contienen
las 5 columnas del 2º semestre que producirían data leakage respecto al target
nota_media_2sem (asignaturas_2sem_evaluadas, asignaturas_2sem_aprobadas,
asignaturas_2sem_sin_evaluacion, asignaturas_2sem_convalidadas,
asignaturas_2sem_matriculadas).

La pregunta que respondemos es: ¿hasta qué punto es posible anticipar el
rendimiento del 2º semestre a partir del perfil de entrada del alumno,
su contexto socioeconómico y su desempeño en el 1er semestre?
"""

print("[ 1/5 ] Cargando datos...")
X_train    = pd.read_csv(f"{DATA_DIR}/X_train_reg.csv")
X_test     = pd.read_csv(f"{DATA_DIR}/X_test_reg.csv")
y_train_df = pd.read_csv(f"{DATA_DIR}/y_train.csv")
y_test_df  = pd.read_csv(f"{DATA_DIR}/y_test.csv")

y_train = y_train_df["nota_media_2sem"].values
y_test  = y_test_df["nota_media_2sem"].values

print(f"        X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"        Target train — media: {y_train.mean():.3f} | std: {y_train.std():.3f}")
print(f"        Target test  — media: {y_test.mean():.3f}  | std: {y_test.std():.3f}")

# ── 2. GRIDSEARCHCV ───────────────────────────────────────────────────────────
"""
Buscamos el alpha óptimo para Ridge y Lasso por separado con GridSearchCV k=5.
Nota importante: en regresión usamos alpha (fuerza de regularización directa),
al contrario que en clasificación donde usábamos C (inversa de la regularización).
Un alpha más grande = regularización más fuerte.

Ridge → penaliza el cuadrado de los coeficientes. Los distribuye entre features
        correlacionadas sin anularlos. Ideal dado que la colinealidad entre
        variables académicas es una realidad ya confirmada en clasificación.

Lasso → penaliza el valor absoluto. Hace selección automática de variables
        anulando las menos informativas. Nos dará riqueza interpretativa extra
        sobre qué features del 1er semestre y del perfil del alumno realmente
        anticipan la nota del 2º.

6 alphas en escala logarítmica × 2 modelos × 5 folds = 60 entrenamientos.
Usamos neg_root_mean_squared_error como scoring (GridSearchCV maximiza,
nosotros queremos minimizar RMSE → usamos la versión negada).
"""

print("\n[ 2/5 ] GridSearchCV (k=5, 12 combinaciones = 60 entrenamientos)...")

alphas     = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {"alpha": alphas}
cv         = KFold(n_splits=5, shuffle=True, random_state=42)

# ── Ridge ──
grid_ridge = GridSearchCV(
    estimator=Ridge(),
    param_grid=param_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0,
    return_train_score=True,
)
grid_ridge.fit(X_train, y_train)

# ── Lasso ──
grid_lasso = GridSearchCV(
    estimator=Lasso(max_iter=10000),
    param_grid=param_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0,
    return_train_score=True,
)
grid_lasso.fit(X_train, y_train)

best_alpha_ridge   = grid_ridge.best_params_["alpha"]
best_alpha_lasso   = grid_lasso.best_params_["alpha"]
best_rmse_ridge_cv = -grid_ridge.best_score_
best_rmse_lasso_cv = -grid_lasso.best_score_

print(f"\n        Ridge — alpha óptimo: {best_alpha_ridge}  |  RMSE (CV): {best_rmse_ridge_cv:.4f}")
print(f"        Lasso — alpha óptimo: {best_alpha_lasso}  |  RMSE (CV): {best_rmse_lasso_cv:.4f}")

# ── 3. REENTRENAR SOBRE TODO EL TRAIN ─────────────────────────────────────────
"""
GridSearchCV con refit=True (por defecto) ya ha reentrenado los mejores modelos
sobre todo X_train. Los extraemos directamente para evaluar en test.
Calculamos también cuántas features ha mantenido Lasso activas para saber
qué fracción del espacio de entrada resulta realmente predictiva.
"""
print("\n[ 3/5 ] Reentrenando mejores modelos sobre todo el train...")
mejor_ridge = grid_ridge.best_estimator_
mejor_lasso = grid_lasso.best_estimator_

n_features_total      = X_train.shape[1]
n_coefs_lasso_activos = int(np.sum(np.abs(mejor_lasso.coef_) > 1e-4))
print(f"        Lasso mantiene {n_coefs_lasso_activos} de {n_features_total} features activas "
      f"({100*n_coefs_lasso_activos/n_features_total:.1f}%)")

# ── 4. EVALUACIÓN SOBRE TEST ──────────────────────────────────────────────────
"""
Evaluamos ambos modelos sobre test con sus hiperparámetros óptimos.
Métricas:
  RMSE → penaliza errores grandes de forma cuadrática. Es la métrica principal
         porque en este contexto equivocarse por 3 puntos es cualitativamente
         peor que equivocarse por 0.5.
  R²   → fracción de la varianza del target explicada por el modelo.
         R²=1 → ajuste perfecto | R²=0 → el modelo no mejora predecir la media.
         Nos da intuición sobre si el 1er semestre realmente anticipa el 2º.
"""
print("\n[ 4/5 ] Evaluando sobre test...")

y_pred_ridge = mejor_ridge.predict(X_test)
rmse_ridge   = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge     = r2_score(y_test, y_pred_ridge)

y_pred_lasso = mejor_lasso.predict(X_test)
rmse_lasso   = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso     = r2_score(y_test, y_pred_lasso)

print(f"\n  ── MÉTRICAS FINALES (TEST) ──────────────────────────────────")
print(f"  {'Modelo':<10}  {'alpha':>8}  {'RMSE (test)':>12}  {'R² (test)':>10}")
print(f"  {'Ridge':<10}  {best_alpha_ridge:>8}  {rmse_ridge:>12.4f}  {r2_ridge:>10.4f}")
print(f"  {'Lasso':<10}  {best_alpha_lasso:>8}  {rmse_lasso:>12.4f}  {r2_lasso:>10.4f}")
print(f"\n  Lasso features activas: {n_coefs_lasso_activos} / {n_features_total}")

# Para las visualizaciones usamos el modelo con menor RMSE en test.
# En caso de empate preferimos Lasso por su mayor interpretabilidad.
if rmse_lasso <= rmse_ridge:
    y_pred_best = y_pred_lasso
    rmse_best   = rmse_lasso
    r2_best     = r2_lasso
    nombre_best = f"Lasso  (alpha={best_alpha_lasso})"
else:
    y_pred_best = y_pred_ridge
    rmse_best   = rmse_ridge
    r2_best     = r2_ridge
    nombre_best = f"Ridge  (alpha={best_alpha_ridge})"

print(f"\n  → Modelo seleccionado para análisis: {nombre_best}")

# ── 5. VISUALIZACIONES ────────────────────────────────────────────────────────
"""
Panel de 4 gráficas diseñadas para extraer conclusiones específicas de este dataset:

  5a. Curva alpha vs RMSE (CV) — igual que en los otros modelos, permite ver
      la sensibilidad de Ridge y Lasso a la regularización y justifica el alpha
      óptimo elegido.

  5b. Predicciones vs valores reales coloreadas por clase del alumno
      (Graduado / Abandono / Matriculado) — responde si el modelo predice
      igual de bien para todos los perfiles. Si Matriculado tiene más dispersión
      sería coherente con lo que ya vimos en clasificación: es una clase de
      tránsito con perfiles académicos menos consolidados.

  5c. Distribución de residuos + scatter residuos vs predichos — diagnóstico
      directo del modelo lineal. Si los residuos tienen asimetría marcada
      o varianza creciente con la nota predicha, el modelo tiene limitaciones
      estructurales que justifican explorar modelos no lineales en el futuro.

  5d. Top 15 coeficientes Lasso con signo — la gráfica más interpretativa:
      qué variables del 1er semestre y del perfil del alumno tienen más peso
      en anticipar la nota del 2º. El signo es clave: coeficiente positivo =
      sube la nota predicha, negativo = la baja.
"""
print("\n[ 5/5 ] Generando visualizaciones...")

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

# ── 5a. Curva alpha vs RMSE (CV) ─────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])

cv_res_ridge = pd.DataFrame(grid_ridge.cv_results_).sort_values("param_alpha")
cv_res_lasso = pd.DataFrame(grid_lasso.cv_results_).sort_values("param_alpha")

rmse_ridge_cv_vals = -cv_res_ridge["mean_test_score"].values
std_ridge_cv_vals  =  cv_res_ridge["std_test_score"].values
rmse_lasso_cv_vals = -cv_res_lasso["mean_test_score"].values
std_lasso_cv_vals  =  cv_res_lasso["std_test_score"].values

x_idx = range(len(alphas))

ax0.plot(x_idx, rmse_ridge_cv_vals, marker="o", color=AZUL,
         label="Ridge", linewidth=2)
ax0.fill_between(x_idx,
                 rmse_ridge_cv_vals - std_ridge_cv_vals,
                 rmse_ridge_cv_vals + std_ridge_cv_vals,
                 alpha=0.15, color=AZUL)

ax0.plot(x_idx, rmse_lasso_cv_vals, marker="s", color=ROJO,
         label="Lasso", linewidth=2)
ax0.fill_between(x_idx,
                 rmse_lasso_cv_vals - std_lasso_cv_vals,
                 rmse_lasso_cv_vals + std_lasso_cv_vals,
                 alpha=0.15, color=ROJO)

ax0.set_xticks(x_idx)
ax0.set_xticklabels([str(a) for a in alphas])
ax0.set_xlabel("Valor de alpha (fuerza de regularización)")
ax0.set_ylabel("RMSE (CV)")
ax0.set_title("Búsqueda de hiperparámetros — GridSearchCV", fontweight="bold")
ax0.legend(fontsize=9, frameon=False)
ax0.yaxis.grid(True, linestyle="--", alpha=0.4)
ax0.set_axisbelow(True)

# Marcar óptimos de cada modelo
best_idx_ridge = alphas.index(best_alpha_ridge)
best_idx_lasso = alphas.index(best_alpha_lasso)
for best_idx, best_rmse, color, label in [
    (best_idx_ridge, best_rmse_ridge_cv, AZUL, f"Ridge óptimo\nalpha={best_alpha_ridge}\nRMSE={best_rmse_ridge_cv:.3f}"),
    (best_idx_lasso, best_rmse_lasso_cv, ROJO, f"Lasso óptimo\nalpha={best_alpha_lasso}\nRMSE={best_rmse_lasso_cv:.3f}"),
]:
    ax0.axvline(best_idx, color=color, linestyle=":", linewidth=1.2, alpha=0.6)
    offset_x = 0.3 if best_idx < len(alphas) - 2 else -1.8
    offset_y = 0.05
    ax0.annotate(label,
                 xy=(best_idx, best_rmse),
                 xytext=(best_idx + offset_x, best_rmse + offset_y),
                 fontsize=7.5, color=color,
                 arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

# ── 5b. Predicciones vs valores reales coloreadas por clase ──────────────────
"""
Coloreamos cada punto según la situación final del alumno (graduado, abandono,
matriculado). Esto nos permite responder si el modelo predice igual de bien
para todos los perfiles o si la clase de tránsito "matriculado" sigue siendo
más difícil también en regresión.
"""
ax1 = fig.add_subplot(gs[0, 1])

clases_unicas  = ["abandono", "graduado", "matriculado"]
colores_clases = [AZUL, VERDE, ROJO]

if "objetivo" in y_test_df.columns:
    clases_test = y_test_df["objetivo"].values
    for cls, col in zip(clases_unicas, colores_clases):
        mask = clases_test == cls
        ax1.scatter(y_test[mask], y_pred_best[mask],
                    alpha=0.35, s=14, color=col, label=cls, edgecolors="none")
    ax1.legend(fontsize=8, frameon=False, title="Clase alumno", title_fontsize=8,
               markerscale=1.5)
else:
    ax1.scatter(y_test, y_pred_best, alpha=0.35, s=14, color=AZUL, edgecolors="none")

lim_min = min(y_test.min(), y_pred_best.min()) - 0.3
lim_max = max(y_test.max(), y_pred_best.max()) + 0.3
ax1.plot([lim_min, lim_max], [lim_min, lim_max],
         color=GRIS, linewidth=1.5, linestyle="--", label="Predicción perfecta")
ax1.set_xlim(lim_min, lim_max)
ax1.set_ylim(lim_min, lim_max)
ax1.set_xlabel("Nota real (nota_media_2sem)")
ax1.set_ylabel("Nota predicha")
ax1.set_title(
    f"Predicciones vs valores reales — {nombre_best}\n"
    f"RMSE={rmse_best:.3f}  |  R²={r2_best:.3f}",
    fontweight="bold", fontsize=10)
ax1.xaxis.grid(True, linestyle="--", alpha=0.3)
ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
ax1.set_axisbelow(True)

# ── 5c. Análisis de residuos ─────────────────────────────────────────────────
"""
Dos subpaneles en uno: distribución de residuos (izquierda) y residuos vs
valores predichos (derecha). Juntos diagnostican si la regresión lineal
es apropiada para este problema:
  - La distribución debe ser aproximadamente normal centrada en 0.
  - El scatter no debe mostrar patrones (cono o curva) → sin heterocedasticidad.
  Si la varianza de los residuos crece con la nota predicha (forma de cono)
  indicaría que el modelo predice bien notas bajas pero mal notas altas,
  algo esperable si los alumnos con mejor rendimiento tienen perfiles más
  heterogéneos difíciles de capturar con variables del 1er semestre.
"""
ax2 = fig.add_subplot(gs[1, 0])

residuos = y_test - y_pred_best
skewness  = pd.Series(residuos).skew()
media_res = residuos.mean()

# Histograma de residuos con curva normal teórica superpuesta
counts, bins, _ = ax2.hist(residuos, bins=40, color=AZUL, alpha=0.70,
                            edgecolor="white", linewidth=0.4,
                            density=True, label="Residuos observados")

# Curva normal teórica (misma media y std que los residuos)
x_norm = np.linspace(residuos.min(), residuos.max(), 200)
from scipy.stats import norm
pdf_norm = norm.pdf(x_norm, loc=media_res, scale=residuos.std())
ax2.plot(x_norm, pdf_norm, color=ROJO, linewidth=2,
         linestyle="-", label="Normal teórica")

ax2.axvline(0,       color=GRIS, linewidth=1.5, linestyle="--", label="Error = 0")
ax2.axvline(media_res, color=VERDE, linewidth=1.4, linestyle="-.",
            label=f"Media residuos = {media_res:.3f}")

ax2.set_xlabel("Residuo (real − predicho)")
ax2.set_ylabel("Densidad")
ax2.set_title(
    f"Distribución de residuos\n"
    f"std={residuos.std():.3f}  |  skewness={skewness:.3f}",
    fontweight="bold", fontsize=10)
ax2.legend(fontsize=8, frameon=False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

# ── 5d. Top coeficientes Lasso con signo ─────────────────────────────────────
"""
Mostramos las 15 features con mayor coeficiente absoluto en el modelo Lasso.
El signo es la información clave:
  Azul (positivo) → la feature sube la nota predicha del 2º semestre.
  Rojo (negativo) → la feature la baja.
Esperamos que nota_media_1sem y asignaturas_1sem_aprobadas sean las más
importantes, y que variables socioeconómicas o de deuda aporten información
complementaria. Si Lasso las mantiene con coeficiente activo, significa que
no eran redundantes con las académicas → tienen poder predictivo propio.
"""
ax3 = fig.add_subplot(gs[1, 1])

feature_names = X_train.columns.tolist()
coefs_lasso   = mejor_lasso.coef_
top_n         = 15
sorted_idx    = np.argsort(np.abs(coefs_lasso))[::-1][:top_n]
top_names     = [feature_names[i] for i in sorted_idx]
top_vals      = coefs_lasso[sorted_idx]

# Ordenamos de mayor a menor valor absoluto (mayor arriba en barh → invertimos)
orden = np.argsort(np.abs(top_vals))           # menor → mayor
top_names_ord = [top_names[i] for i in orden]
top_vals_ord  = [top_vals[i]  for i in orden]
colores_coef  = [AZUL if v >= 0 else ROJO for v in top_vals_ord]

bars = ax3.barh(range(top_n), top_vals_ord,
                color=colores_coef, alpha=0.85, edgecolor="white")
ax3.axvline(0, color=GRIS, linewidth=1.0, linestyle="--")
ax3.set_yticks(range(top_n))
ax3.set_yticklabels(top_names_ord, fontsize=8)
ax3.set_xlabel(f"Coeficiente Lasso  (alpha={best_alpha_lasso})")
ax3.set_title(
    f"Top {top_n} coeficientes — Lasso\n"
    "(azul = sube nota predicha | rojo = baja nota predicha)",
    fontweight="bold", fontsize=10)
ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)

# Añadir valor numérico al final de cada barra
for bar, val in zip(bars, top_vals_ord):
    offset = 0.002 if val >= 0 else -0.002
    ha     = "left" if val >= 0 else "right"
    ax3.text(val + offset, bar.get_y() + bar.get_height() / 2,
             f"{val:+.3f}", va="center", ha=ha, fontsize=7.5, color=GRIS)

# ── Título general ────────────────────────────────────────────────────────────
fig.suptitle(
    f"Regresión Lineal — Predicción de nota_media_2sem  |  "
    f"Mejor modelo: {nombre_best}  |  RMSE={rmse_best:.3f}  |  R²={r2_best:.3f}",
    fontsize=12, fontweight="bold", y=1.01
)

plt.savefig(f"{FIGURES_DIR}/fig_linear_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Guardado: {FIGURES_DIR}/fig_linear_regression.png")

# ── FIGURA EXTRA: Diagnóstico de métricas y límite del modelo ─────────────────
"""
Esta figura responde dos preguntas que el panel principal deja abiertas:

  Fig A (tabla de métricas): ¿cuánto mejor o peor es Ridge vs Lasso en test?
  RMSE, MAE y R² lado a lado. El MAE es útil aquí porque es menos sensible
  a los outliers (los ceros de abandono) y da una idea más honesta del error
  típico del modelo sobre los alumnos "normales".

  Fig B (residuos vs predichos): ¿dónde falla el modelo? Un patrón de cono
  (varianza creciente con la predicción) revelaría heterocedasticidad. Un
  cluster de puntos con residuo muy negativo en predicciones bajas delataría
  el problema de los alumnos con nota real 0 que el modelo sobreestima.

  Fig C (nota_1sem por clase): la gráfica clave para entender por qué el modelo
  no puede anticipar los abandonos con nota real 0. Si los alumnos de abandono
  tienen nota_media_1sem alta con frecuencia, no existe señal en el 1er semestre
  que permita distinguirlos de los graduados. El modelo asigna una nota alta
  porque la información disponible lo indica, y falla porque el abandono fue
  una decisión posterior no capturada por ninguna variable del dataset.
  Esto no es un fallo del modelo sino un límite intrínseco del problema.
"""
print("\n[ 6/5 ] Generando figura de diagnóstico extra...")

from sklearn.metrics import mean_absolute_error

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

fig2 = plt.figure(figsize=(18, 6))
gs2  = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.42)

# ── Fig A: Tabla de métricas Ridge vs Lasso ───────────────────────────────────
ax_a = fig2.add_subplot(gs2[0])

modelos    = ["Ridge", "Lasso"]
rmse_vals  = [rmse_ridge,  rmse_lasso]
mae_vals   = [mae_ridge,   mae_lasso]
r2_vals    = [r2_ridge,    r2_lasso]
alpha_vals = [best_alpha_ridge, best_alpha_lasso]

x      = np.arange(3)           # 3 métricas
ancho  = 0.30
offset = [-0.15, 0.15]
colores_mod = [AZUL, ROJO]

for i, (mod, rmse_v, mae_v, r2_v, col) in enumerate(
        zip(modelos, rmse_vals, mae_vals, r2_vals, colores_mod)):
    vals = [rmse_v, mae_v, r2_v]
    bars = ax_a.bar(x + offset[i], vals, ancho, label=mod,
                    color=col, alpha=0.82, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.15,
                  f"{val:.3f}", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color=col)

ax_a.set_xticks(x)
ax_a.set_xticklabels(["RMSE", "MAE", "R²"], fontsize=11)
ax_a.set_ylabel("Valor")
ax_a.set_title("Comparativa de métricas en test\nRidge vs Lasso",
               fontweight="bold")
ax_a.legend(fontsize=9, frameon=False)
ax_a.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_a.set_axisbelow(True)

# Anotar alphas óptimos debajo de los nombres de modelo
for i, (mod, alpha_v, col) in enumerate(zip(modelos, alpha_vals, colores_mod)):
    ax_a.annotate(f"(alpha={alpha_v})",
                  xy=(0.25 + i * 0.5, -0.07), xycoords="axes fraction",
                  ha="center", fontsize=8, color=col)

# ── Fig B: Residuos vs predichos (heterocedasticidad) ────────────────────────
ax_b = fig2.add_subplot(gs2[1])

residuos_best = y_test - y_pred_best

if "objetivo" in y_test_df.columns:
    clases_test = y_test_df["objetivo"].values
    for cls, col in zip(clases_unicas, colores_clases):
        mask = clases_test == cls
        ax_b.scatter(y_pred_best[mask], residuos_best[mask],
                     alpha=0.30, s=13, color=col, label=cls, edgecolors="none")
    ax_b.legend(fontsize=8, frameon=False, title="Clase alumno",
                title_fontsize=8, markerscale=1.5)
else:
    ax_b.scatter(y_pred_best, residuos_best,
                 alpha=0.30, s=13, color=AZUL, edgecolors="none")

ax_b.axhline(0, color=GRIS, linewidth=1.5, linestyle="--")
ax_b.set_xlabel("Nota predicha")
ax_b.set_ylabel("Residuo (real − predicho)")
ax_b.set_title(
    "Residuos vs valores predichos\n"
    "(patrón de cono → heterocedasticidad | cluster bajo → fallo en abandonos)",
    fontweight="bold", fontsize=10)
ax_b.xaxis.grid(True, linestyle="--", alpha=0.3)
ax_b.yaxis.grid(True, linestyle="--", alpha=0.3)
ax_b.set_axisbelow(True)

# ── Fig C: Distribución nota_1sem por clase ───────────────────────────────────
"""
Si los abandonos tienen nota_media_1sem similar a la de los graduados, el modelo
no tiene ninguna señal que le permita diferenciarlos y predecir nota=0 en el 2º.
El solapamiento entre las distribuciones de abandono y graduado cuantifica
exactamente cuán imposible es ese problema desde la información disponible.
"""
ax_c = fig2.add_subplot(gs2[2])

# Necesitamos la nota_media_1sem de train+test con sus clases
# La reconstruimos desde los ficheros completos de clasificación
try:
    X_full_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_full_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    X_full = pd.concat([X_full_train, X_full_test], ignore_index=True)

    y_full_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")
    y_full_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv")
    y_full = pd.concat([y_full_train, y_full_test], ignore_index=True)

    col_nota1 = "nota_media_1sem"
    if col_nota1 in X_full.columns and "objetivo" in y_full.columns:
        for cls, col in zip(clases_unicas, colores_clases):
            datos_cls = X_full.loc[y_full["objetivo"] == cls, col_nota1]
            ax_c.hist(datos_cls, bins=35, alpha=0.55, color=col,
                      edgecolor="white", linewidth=0.4,
                      density=True, label=f"{cls}  (n={len(datos_cls)})")
            # Mediana como línea vertical
            mediana = datos_cls.median()
            ax_c.axvline(mediana, color=col, linewidth=1.6,
                         linestyle="--", alpha=0.9)

        ax_c.set_xlabel("nota_media_1sem (estandarizada tras preprocesado)")
        ax_c.set_ylabel("Densidad")
        ax_c.set_title(
            "Distribución de nota_media_1sem por clase\n"
            "¿Por qué el modelo no anticipa los abandonos con nota real 0?",
            fontweight="bold", fontsize=10)
        ax_c.legend(fontsize=8.5, frameon=False)
        ax_c.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax_c.set_axisbelow(True)

        # Anotación explicativa
        ax_c.annotate(
            "Si abandono y graduado se solapan\naquí, el 1er semestre no\n"
            "distingue quién va a abandonar",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=8, color=GRIS,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GRIS, alpha=0.7)
        )
    else:
        ax_c.text(0.5, 0.5, "nota_media_1sem no disponible\nen X_train.csv",
                  ha="center", va="center", transform=ax_c.transAxes, color=GRIS)

except FileNotFoundError:
    ax_c.text(0.5, 0.5, "X_train.csv / X_test.csv\nno encontrados",
              ha="center", va="center", transform=ax_c.transAxes, color=GRIS)

fig2.suptitle(
    "Diagnóstico del modelo — ¿Dónde y por qué falla la regresión lineal?",
    fontsize=13, fontweight="bold", y=1.02
)

plt.savefig(f"{FIGURES_DIR}/fig_linear_regression_diagnostico.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Guardado: {FIGURES_DIR}/fig_linear_regression_diagnostico.png")
print("\n[ DONE ] Regresión lineal completada.")
