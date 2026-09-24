"""
Genera los ficheros listos para modelado:
    - X_train.csv / X_test.csv          → features completas (clasificación + no supervisado)
    - X_train_reg.csv / X_test_reg.csv  → features sin leakage (regresión)
    - y_train.csv / y_test.csv          → targets: 'objetivo' + 'nota_media_2sem'

Pasos aplicados (en orden):
    1. Carga del dataset
    2. Agrupación semántica de variables de alta cardinalidad
    3. Label Encoding de variables binarias
    4. One-Hot Encoding de variables nominales (drop_first=True)
    5. Transformación Yeo-Johnson en variables asimétricas (|skew| > 1)
    6. Split estratificado 70% train / 30% test (random_state=42)
    7. StandardScaler (fit en train, transform en train y test)
    8. Guardado de ficheros CSV
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
import os


# CONSTANTES
RANDOM_STATE = 42
TEST_SIZE    = 0.30   # 70% train, 30% test
DATA_PATH    = "rendimiento_estudiantes.csv"
OUTPUT_DIR   = "data_processed"

# Variables del 2º semestre que causan leakage en regresión
LEAKAGE_COLS = [
    "asignaturas_2sem_evaluadas",
    "asignaturas_2sem_aprobadas",
    "asignaturas_2sem_sin_evaluacion",
    "asignaturas_2sem_convalidadas",
    "asignaturas_2sem_matriculadas",
] 

# Variables binarias que vamos a guardar como 0 o 1
BINARY_COLS = [
    "desplazado",
    "necesidades_educativas_especiales",
    "deudor",
    "matricula_al_dia",
    "genero",
    "becado",
    "internacional",
    "asistencia_diurna_vespertina",
]

# Variables nominales de no tienen tantas categorías, haremos One-Hot
OHE_COLS = [
    "estado_civil",
    "modo_solicitud",
    "curso",
    "cualificacion_previa",
]

# Variables con muchas categorías, agruparemos semánticamente en el siguiente paso
HIGH_CARD_COLS = [
    "cualificacion_madre",
    "cualificacion_padre",
    "ocupacion_madre",
    "ocupacion_padre",
    "nacionalidad",
]

# Variables numéricas continuas con |skew| > 1 → Yeo-Johnson ya que pueden tener 0 y por ello Box-Cox no sirve
SKEWED_COLS = [
    "nota_media_1sem",
    "nota_media_2sem",
]
#"edad_al_matricularse" antes estaba dentro de la lista, leyendo la memoria se entenderá porque hemos decidido dejarlo fuera


# PASO 2: AGRUPACIONES SEMÁNTICAS
"""
Con estas funciones podremos reducir columnas con muchas categorías que son susceptibles de dar
muchos problemas al codificar ya que aumentarían mucho la dimensión del input y podríamos perder
estabilidad en el modelo.
"""

def agrupar_cualificacion(valor):
    """
    Agrupa las 29–34 categorías de cualificacion_madre/padre en 5 niveles educativos.
    Agrupamos por nivel de formación académica de menor a mayor.
    """
    v = str(valor).lower()
    if any(x in v for x in ["no_sabe", "sabe_leer", "desconocido"]):
        return "sin_estudios" #padre o madre sin estudios
    elif any(x in v for x in ["1er_ciclo", "2o_ciclo", "3er_ciclo",
                               "7o_ano", "8o_ano", "9o_ano"]):
        return "primaria" #padre o madre con educación primaria solo
    elif any(x in v for x in ["10o_ano", "11o_ano", "12o_ano", "otro_11",
                               "bachillerato", "curso_tecnico_profesional",
                               "curso_general", "curso_complementario",
                               "contabilidad", "2o_ano_complementario"]):
        return "secundaria" #padre o madre con educación secundaria solo
    elif any(x in v for x in ["tecnico_superior", "especializacion_tecnologica",
                               "frecuencia_de_educacion_superior",
                               "curso_superior_especializado"]):
        return "formacion_profesional" #padre o madre con formación profesional
    elif "educacion_superior" in v: 
        return "superior"#padre o madre con educación superior
    else:
        return "secundaria"   #en caso de fallo lo guardamos como secundaria 


def agrupar_ocupacion(valor):
    """
    Agrupa las 32–46 categorías de ocupacion_madre/padre en 6 grupos.
    Vamos a usar como criterio la clasificación internacional de ocupaciones
    """
    v = str(valor).lower()
    if any(x in v for x in ["en_blanco", "otra_situacion", "estudiante",
                              "no_cualificad"]):
        return "no_cualificados_otros"
    elif any(x in v for x in ["director", "representante", "legislativo",
                               "gerente", "ejecutivo"]):
        return "directivos_profesionales"
    elif any(x in v for x in ["especialista", "docente", "profesional_de_la_salud",
                               "profesionales_de_la_salud", "intelectual",
                               "cientifico", "especialistas_en_tic",
                               "finanzas", "contabilidad_y_organ",
                               "especialistas_en_finanzas",
                               "especialistas_en_ciencias"]):
        return "directivos_profesionales"
    elif any(x in v for x in ["tecnic", "intermedi", "nivel_intermedio"]):
        return "tecnicos_intermedios"
    elif any(x in v for x in ["administrativ", "oficina", "secretari",
                               "operador", "datos_contabilidad"]):
        return "administrativos"
    elif any(x in v for x in ["servicio", "vendedor", "cuidador",
                               "seguridad", "limpieza", "personal_de_proteccion",
                               "ayudante", "comida"]):
        return "servicios_comercio"
    elif any(x in v for x in ["agricultor", "pescador", "silvicultura",
                               "ganadero", "recolector", "agropecuario"]):
        return "trabajadores_manuales"
    elif any(x in v for x in ["construccion", "industria", "metalurgia",
                               "electricidad", "electronica", "conductor",
                               "maquinaria", "montador", "instalacion",
                               "artesania", "impresion", "confeccion",
                               "alimentaria", "madera", "manufactura",
                               "extractiva", "transporte", "cualificad"]):
        return "trabajadores_manuales"
    elif any(x in v for x in ["fuerzas_armadas", "oficial", "sargento",
                               "proteccion"]):
        return "directivos_profesionales"
    else:
        return "servicios_comercio"   # fallback conservador


def agrupar_nacionalidad(valor):
    """
    Agrupa las 21 nacionalidades en 3 bloques geográfico-culturales.
    Agruparemos por cercanía al sistema educativo portugues.
    """
    PORTUGUESA = {"portuguesa"}
    EUROPA = {"espanola", "alemana", "inglesa", "italiana", "neerlandesa",
              "rumana", "rusa", "lituana", "moldava", "ucraniana", "turca"}
    # el resto → africa_latam
    v = str(valor).lower()
    if v in PORTUGUESA:
        return "portuguesa"
    elif v in EUROPA:
        return "europa_otros"
    else:
        return "africa_latam"



# FUNCIÓN PRINCIPAL

def run_preprocessing(data_path=DATA_PATH, output_dir=OUTPUT_DIR, verbose=True):
    """
    Ejecuta el preprocesado y guarda los ficheros CSV.

    Returns
    
    dict con claves:
        X_train, X_test               → DataFrames con features completas
        X_train_reg, X_test_reg       → DataFrames sin leakage (regresión)
        y_train, y_test               → DataFrames con objetivo + nota_media_2sem
        feature_names                 → lista de columnas de X_train
        scaler                        → StandardScaler ajustado
        power_transformer             → PowerTransformer (Yeo-Johnson) ajustado
        
    Es clave devolver el scaler y el power_transformer para poder posteriormente deshacer las transformaciones
    y hacer interpretación con los valores reales.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── PASO 1: Carga ────────────────────────────────────────────────────────
    """
    Cargamos el dataset y verificamos si hay valores nulos. Al no haberlos no haremos imputación
    """
    if verbose: print("[ 1/7 ] Cargando dataset...")
    df = pd.read_csv(data_path, sep=";")
    if verbose: print(f"        Shape: {df.shape} | Nulos: {df.isnull().sum().sum()}")
    
    # Trás ver los resultados sin estas lineas de código consideramos que es necesario eliminar los outliers de la variable problemática
    # Se eliminan los estudiantes con edad > Q3 + 1.5*IQR (34 años)
    q3  = df["edad_al_matricularse"].quantile(0.75)
    iqr = df["edad_al_matricularse"].quantile(0.75) - df["edad_al_matricularse"].quantile(0.25)
    lim = q3 + 1.5 * iqr
    n_antes = len(df)
    df = df[df["edad_al_matricularse"] <= lim].reset_index(drop=True)
    if verbose: print(f"        Outliers edad eliminados: {n_antes - len(df)} filas (>{lim:.0f} años) → shape: {df.shape}")


    # ── PASO 2: Agrupación semántica de alta cardinalidad ────────────────────
    """
    Aquí reducimos la alta cardinalidad de esas columnas que hemos detectado como problemáticas
    al tener tantas categorías. De esta manera, cuando hagamos one-hot no obtendremos tantas 
    dimensiones añadidas.
    """
    if verbose: print("[ 2/7 ] Agrupando variables de alta cardinalidad...")

    df["cualificacion_madre"] = df["cualificacion_madre"].apply(agrupar_cualificacion)
    df["cualificacion_padre"] = df["cualificacion_padre"].apply(agrupar_cualificacion)
    df["ocupacion_madre"]     = df["ocupacion_madre"].apply(agrupar_ocupacion)
    df["ocupacion_padre"]     = df["ocupacion_padre"].apply(agrupar_ocupacion)
    df["nacionalidad"]        = df["nacionalidad"].apply(agrupar_nacionalidad)

    if verbose:
        for col in HIGH_CARD_COLS: #demostramos la reducción de categorías en las columnas
            print(f"        {col}: {df[col].nunique()} grupos → {sorted(df[col].unique())}")

    # ── PASO 3: Label Encoding de binarias ───────────────────────────────────
    """
    Ahora nos ocupamos de hacer encoding de las variables con dos categorías (binarias)
    Simplemente hacemos un mapeo de las categorías a un cierto número 0 o 1.
    """
    if verbose: print("[ 3/7 ] Label Encoding de variables binarias...")

    binary_maps = {
        "desplazado":                         {"si": 1, "no": 0},
        "necesidades_educativas_especiales":  {"si": 1, "no": 0},
        "deudor":                             {"si": 1, "no": 0},
        "matricula_al_dia":                   {"si": 1, "no": 0},
        "genero":                             {"hombre": 1, "mujer": 0},
        "becado":                             {"si": 1, "no": 0},
        "internacional":                      {"si": 1, "no": 0},
        "asistencia_diurna_vespertina":       {"diurna": 1, "vespertina": 0},
    }
    for col, mapping in binary_maps.items():
        df[col] = df[col].map(mapping)
        if verbose: print(f"        {col}: {mapping}")

    # ── PASO 4: One-Hot Encoding ─────────────────────────────────────────────
    """
    Aquí, vamos a codificar ya todas las variables con más de una categoría. Juntamos
    las columnas que ya tenían cardinalidad válida para la codificación con las columnas
    que hemos reducido manualmente. 
    Creamos las columnas codificadas y mostramos por pantalla cómo la dimensión del input
    ha aumentado considerablemente.
    """
    
    if verbose: print("[ 4/7 ] One-Hot Encoding (drop_first=True)...")

    all_ohe_cols = OHE_COLS + HIGH_CARD_COLS
    df = pd.get_dummies(df, columns=all_ohe_cols, drop_first=True, dtype=int)
    
    if verbose: print(f"        Columnas tras OHE: {df.shape[1]}")

    # ── PASO 5: Separar targets y features ───────────────────────────────────
    """
    Aquí separamos los features X de las variables que servirán en algún modelo 
    como target y. Hacemos coding con LabelEncoder de la variable objetivo y 
    generamos el conjunto X_reg sin las colmunas susceptibles de producir
    Data Leakage.
    """
    if verbose: print("[ 5/7 ] Separando targets y features...")

    y = df[["objetivo", "nota_media_2sem"]].copy()

    # Encode target de clasificación
    le = LabelEncoder()
    y["objetivo_encoded"] = le.fit_transform(y["objetivo"])
    if verbose: print(f"        Clases: {list(le.classes_)} → {list(range(len(le.classes_)))}")

    # Features: quitamos ambos targets del X completo
    X = df.drop(columns=["objetivo"])

    # Features para regresión: además quitamos las columnas de leakage
    X_reg = X.drop(columns=LEAKAGE_COLS+["nota_media_2sem"])
    if verbose:
        print(f"        X completo:   {X.shape[1]} features")
        print(f"        X regresión:  {X_reg.shape[1]} features (sin {len(LEAKAGE_COLS)} leakage)")

    # ── PASO 6: Split estratificado 70/30 ────────────────────────────────────
    """
    Separamos los conjuntos X e y en train y test. El conjunto de validación lo gestionaremos ya cuando
    entrenemos los modelos y usemos cross validation. Además generamos los conjuntos de train y test
    específicos para regresión sin las columnas problemáticas.
    """
    if verbose: print("[ 6/7 ] Split estratificado 70/30 (stratify=objetivo)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y["objetivo"]
    )
    X_train_reg = X_train[X_reg.columns]
    X_test_reg  = X_test[X_reg.columns]

    if verbose:
        print(f"        Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")
        for cls in y_train["objetivo"].unique():
            n = (y_train["objetivo"] == cls).sum()
            pct = n / len(y_train) * 100
            print(f"        Train → {cls}: {n} ({pct:.1f}%)")

    # ── PASO 7: Yeo-Johnson en variables asimétricas ─────────────────────────
    """
    Aplicamos la transformacíon Yeo-Johnson para las tres variables que hemos
    considerado cómo suficientemente asimétricas para ser problemáticas.
    Generamos la transformación sobre train y aplicamos luego en test.
    """
    if verbose: print("[ 7a/7 ] Yeo-Johnson en variables asimétricas...")

    skewed_in_X = [c for c in SKEWED_COLS if c in X_train.columns]

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_train[skewed_in_X] = pt.fit_transform(X_train[skewed_in_X])
    X_test[skewed_in_X]  = pt.transform(X_test[skewed_in_X])
    # Solo copiamos a X_reg las columnas transformadas que realmente están en X_reg
    # (nota_media_2sem NO debe entrar en X_reg aunque esté en skewed_in_X)
    skewed_in_X_reg = [c for c in skewed_in_X if c in X_train_reg.columns]
    X_train_reg[skewed_in_X_reg] = X_train[skewed_in_X_reg]
    X_test_reg[skewed_in_X_reg]  = X_test[skewed_in_X_reg]

    # Actualizamos nota_media_2sem en y con la versión ya transformada por Yeo-Johnson,
    # que es la que estará en el mismo espacio que las features de X_train.
    # Esto garantiza consistencia entre el target de regresión y el espacio de features.
    if "nota_media_2sem" in X_train.columns:
        y_train["nota_media_2sem"] = X_train["nota_media_2sem"].values
        y_test["nota_media_2sem"]  = X_test["nota_media_2sem"].values

    if verbose:
        skews_after = X_train[skewed_in_X].skew()
        for col, sk in skews_after.items():
            print(f"        {col}: skew tras YJ = {sk:.3f}")

    # ── PASO 7b: StandardScaler ───────────────────────────────────────────────
    """
    Estandarizamos todas las variables continuas teniendo siempre en cuenta que se 
    genera la estandarización sobre train y luego la aplicamos sobre test
    """
    if verbose: print("[ 7b/7 ] StandardScaler (fit en train)...")

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    num_cols_reg = X_train_reg.select_dtypes(include=[np.number]).columns.tolist()
    scaler_reg = StandardScaler()
    X_train_reg[num_cols_reg] = scaler_reg.fit_transform(X_train_reg[num_cols_reg])
    X_test_reg[num_cols_reg]  = scaler_reg.transform(X_test_reg[num_cols_reg])

    if verbose: print(f"        Escaladas {len(num_cols)} columnas numéricas")

    # GUARDADO
    """
    Por último guardamos los conjuntos listos para el posterior uso en los modelos el directorio
    de data_processed.
    """
    if verbose: print("\n[ OK ] Guardando ficheros en '{}'...".format(output_dir))

    X_train.to_csv(f"{output_dir}/X_train.csv",     index=False)
    X_test.to_csv( f"{output_dir}/X_test.csv",      index=False)
    X_train_reg.to_csv(f"{output_dir}/X_train_reg.csv", index=False)
    X_test_reg.to_csv( f"{output_dir}/X_test_reg.csv",  index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv",     index=False)
    y_test.to_csv( f"{output_dir}/y_test.csv",      index=False)

    if verbose:
        print(f"\n  X_train.csv      → {X_train.shape}")
        print(f"  X_test.csv       → {X_test.shape}")
        print(f"  X_train_reg.csv  → {X_train_reg.shape}")
        print(f"  X_test_reg.csv   → {X_test_reg.shape}")
        print(f"  y_train.csv      → {y_train.shape}  (cols: {list(y_train.columns)})")
        print(f"  y_test.csv       → {y_test.shape}")
        print("\n  Preprocesado completado correctamente.")

    return {
        "X_train":          X_train,
        "X_test":           X_test,
        "X_train_reg":      X_train_reg,
        "X_test_reg":       X_test_reg,
        "y_train":          y_train,
        "y_test":           y_test,
        "feature_names":    list(X_train.columns),
        "label_encoder":    le,
        "scaler":           scaler,
        "scaler_reg":       scaler_reg,
        "power_transformer":pt,
        "skewed_cols_X":    skewed_in_X,
    }



if __name__ == "__main__":
    run_preprocessing()
