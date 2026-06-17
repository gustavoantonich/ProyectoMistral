# -*- coding: utf-8 -*-
# SportCars_RAG_MultiAgente.py
# Converted from SportCars_RAG_MultiAgente.ipynb


# ==========================================================
# # **SISTEMA RAG MULTI-AGENTE CON TRANSFORMERS Y MISTRAL**
# ### **Dataset: Edmunds Consumer Car Ratings & Reviews (Kaggle)**
# 
# ---
# 
# ## **ARQUITECTURA DEL NOTEBOOK (5 AGENTES)**
# 
# | Agente | Rol | Responsabilidad |
# |--------|-----|-----------------|
# | **AGENTE 1** | Data Analyst | Define y fundamenta la elección del dataset |
# | **AGENTE 2** | Data Engineer | Carga, analiza, normaliza y prepara los datos |
# | **AGENTE 2B** | Trainer | Entrena y optimiza modelo de predicción de rating |
# | **AGENTE 3** | NLP/ML Engineer | Aplica transformers (sentiment + embeddings), construye el RAG |
# | **AGENTE 3B** | Comunicador | Genera reporte ejecutivo en lenguaje natural con Mistral |
# | **AGENTE 4** | AI Agent Architect | Implementa el agente autónomo con function calling |
# 
# ---
# ==========================================================


# ==========================================================
# # ==========================================
# # AGENTE 1: DEFINICIÓN Y FUNDAMENTACIÓN DEL DATASET
# # ==========================================
# 
# ## **Dataset: Edmunds Consumer Car Ratings and Reviews**
# 
# ### **Origen**
# Kaggle — [ankkur13/edmundsconsumer-car-ratings-and-reviews](https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews)
# 
# ### **¿Por qué este dataset?**
# - **Texto enriquecido**: reseñas de consumidores reales sobre autos → ideal para embeddings semánticos y RAG
# - **Datos estructurados**: rating numérico (1-5) → perfecto para el agente calculadora
# - **62 marcas** incluyendo deportivas: Ferrari, Lamborghini, Porsche, McLaren, Aston Martin, Maserati, Lotus, etc.
# - **Normalizable**: requiere limpieza de texto, estandarización de fechas, manejo de nulos y extracción de marca/modelo/año
# - **Mediano-grande**: ~10K-15K reseñas por archivo, dataset completo de ~500K+ filas
# - **Diferente del anterior**: antes trabajamos con ventas transaccionales, ahora con reseñas textuales de autos
# 
# ### **Columnas del dataset**
# | Columna | Tipo | Descripción |
# |---------|------|-------------|
# | `Review_Date` | Texto | Fecha de la reseña (requiere parseo) |
# | `Author_Name` | Texto | Nombre del autor (muchos nulos) |
# | `Vehicle_Title` | Texto | Título completo del vehículo (año, marca, modelo, trim) |
# | `Review_Title` | Texto | Título/resumen de la reseña |
# | `Review` | Texto | **Cuerpo completo de la reseña** → input principal para NLP |
# | `Rating` | Float | Puntuación del consumidor (1.0 - 5.0) |
# 
# ### **Estructura del dataset en Kaggle**
# El dataset contiene archivos CSV separados por marca: `Scraped_Car_Review_{marca}.csv`
# 
# ---
# ==========================================================


# ==========================================================
# # ==========================================
# # AGENTE 2: INGENIERÍA DE DATOS
# # (Carga, Análisis Exploratorio y Normalización)
# # ==========================================
# ==========================================================


# ==========================================================
# ### **FASE 2.1: INSTALACIÓN DE DEPENDENCIAS**
# ==========================================================

# Instalación silenciosa de todas las dependencias del ecosistema
!pip install -q kagglehub pandas numpy matplotlib seaborn
!pip install -q transformers sentence-transformers
!pip install -q langchain langchain-community langchain-mistralai faiss-cpu
!pip install -q mistralai

print("[ENTORNO] Todas las dependencias instaladas correctamente.")

# ==========================================================
# ### **FASE 2.2: DESCARGA DEL DATASET DESDE KAGGLE**
# 
# Usamos `kagglehub` para descargar el dataset directamente. Esto requiere autenticación:
# 1. Ve a tu cuenta de Kaggle → Settings → Create API Token
# 2. Sube `kaggle.json` a Colab o configura las variables de entorno
# ==========================================================

import os
import kagglehub
import pandas as pd
import glob

# Configurar credenciales de Kaggle (requerido para kagglehub)
# En Colab puedes subir tu kaggle.json o configurar las variables:
# from google.colab import files
# files.upload()  # Subir kaggle.json
# os.makedirs('/root/.kaggle', exist_ok=True)
# !mv kaggle.json /root/.kaggle/
# !chmod 600 /root/.kaggle/kaggle.json

print("[KAGGLE] Descargando dataset: ankkur13/edmundsconsumer-car-ratings-and-reviews...")

try:
    path = kagglehub.dataset_download("ankkur13/edmundsconsumer-car-ratings-and-reviews")
    print(f"[KAGGLE] Dataset descargado en: {path}")
except Exception as e:
    print(f"[ERROR] No se pudo descargar desde Kaggle: {e}")
    print("[FALLBACK] Asegúrate de tener configurado kaggle.json o sube manualmente los CSVs.")
    path = None

# ==========================================================
# ### **FASE 2.3: CARGA Y COMBINACIÓN DE ARCHIVOS**
# 
# Cargamos todos los archivos CSV (uno por marca) y los combinamos en un solo DataFrame añadiendo la columna `Brand`.
# ==========================================================

import os
import kagglehub
import pandas as pd
import glob

def cargar_todas_las_marcas(ruta_base):
    """
    Busca todos los archivos Scraped_Car_Review_*.csv en el directorio
    y los combina en un solo DataFrame, extrayendo la marca del nombre del archivo.
    """
    if ruta_base is None:
        print("[ERROR] No hay ruta de dataset disponible.")
        return None

    patron = os.path.join(ruta_base, "Scraped_Car_Review_*.csv")
    archivos = glob.glob(patron)

    print(f"[CARGA] Encontrados {len(archivos)} archivos CSV (una por marca).")

    if not archivos:
        print("[ERROR] No se encontraron archivos CSV en la ruta.")
        return None

    dataframes = []
    for i, archivo in enumerate(archivos):
        nombre_base = os.path.basename(archivo)
        # Extraer marca del nombre: 'Scraped_Car_Review_ferrari.csv' -> 'ferrari'
        marca = nombre_base.replace("Scraped_Car_Review_", "").replace(".csv", "")

        try:
            # Usar engine='python' para mayor robustez con CSVs complejos
            # Se elimina encoding='unicode_escape' ya que puede interferir con engine='python'
            df_temp = pd.read_csv(archivo, engine='python')
            df_temp['Brand'] = marca
            dataframes.append(df_temp)
        except Exception as e:
            print(f"  [AVISO] Error al leer {nombre_base}: {e}")

    df_completo = pd.concat(dataframes, ignore_index=True)
    print(f"[CARGA] Dataset combinado: {df_completo.shape[0]} filas x {df_completo.shape[1]} columnas")
    print(f"[CARGA] Marcas únicas cargadas: {df_completo['Brand'].nunique()}")

    return df_completo


df = cargar_todas_las_marcas(path)

if df is not None:
    print("\n[VISTA PREVIA] Primeras 3 filas:")
    display(df.head(3))

# ==========================================================
# ### **FASE 2.4: ANÁLISIS EXPLORATORIO (EDA)**
# 
# Diagnosticamos la estructura del dataset antes de normalizar.
# ==========================================================

def analisis_exploratorio(df):
    """
    Realiza un diagnóstico completo del dataset:
    - Shape, tipos de datos, nulos
    - Estadísticas descriptivas
    - Distribución de ratings y marcas
    - Métricas textuales para chunking
    """
    print("="*60)
    print("           ANÁLISIS EXPLORATORIO DEL DATASET")
    print("="*60)

    # 1. Estructura general
    print(f"\n[1] DIMENSIONES: {df.shape[0]} filas x {df.shape[1]} columnas\n")

    # 2. Mapeo de tipos y nulos
    info_cols = pd.DataFrame({
        'Tipo': df.dtypes,
        'No_Nulos': df.count(),
        'Nulos': df.isnull().sum(),
        '%_Nulos': (df.isnull().sum() / len(df)) * 100
    })
    print("[2] MAPEO DE COLUMNAS (tipos + nulos):")
    display(info_cols)

    # 3. Distribución de ratings
    print("\n[3] DISTRIBUCIÓN DE RATINGS:")
    print(df['Rating'].describe())

    # 4. Top 10 marcas con más reseñas
    print("\n[4] TOP 10 MARCAS CON MÁS RESEÑAS:")
    top_marcas = df['Brand'].value_counts().head(10)
    print(top_marcas)

    # 5. Métricas textuales (crítico para definir chunk size)
    df['_texto_completo'] = df.apply(
        lambda r: f"{str(r.get('Review_Title', ''))} {str(r.get('Review', ''))}", axis=1
    )
    longitudes = df['_texto_completo'].str.len()
    print(f"\n[5] MÉTRICAS TEXTUALES PARA CHUNKING:")
    print(f"  Longitud promedio (chars): {longitudes.mean():.1f}")
    print(f"  Longitud mediana (chars):  {longitudes.median():.1f}")
    print(f"  Tokens estimados (prom/4): {longitudes.mean() / 4:.1f}")
    print(f"  Reviews vacías: {(longitudes == 0).sum()}")

    # 6. Marcas deportivas disponibles
    marcas_deportivas = ['ferrari', 'lamborghini', 'porsche', 'mclaren',
                         'aston martin', 'maserati', 'lotus', 'bugatti']
    disponibles = [m for m in marcas_deportivas if m in df['Brand'].str.lower().unique()]
    print(f"\n[6] MARCAS DEPORTIVAS DETECTADAS: {disponibles}")

    return df


if df is not None:
    df = analisis_exploratorio(df)

# ==========================================================
# ### **FASE 2.5: NORMALIZACIÓN Y LIMPIEZA**
# 
# Aplicamos las transformaciones necesarias para dejar el dataset listo para NLP:
# ==========================================================

import pandas as pd
import re # Import regex module for more robust string operations

def normalizar_dataset(df):
    """
    Pipeline completo de normalización:
    1. Limpieza de espacios en encabezados
    2. Parseo de fechas
    3. Extracción de año, marca y modelo desde Vehicle_Title
    4. Limpieza de texto en reseñas
    5. Manejo de nulos
    6. Estandarización de categorías
    7. Filtrado de registros válidos
    """
    df_clean = df.copy()

    print("="*60)
    print("           NORMALIZACIÓN DEL DATASET")
    print("="*60)
    print(f"[INICIO] Dataset recibido para normalización: {df_clean.shape[0]} filas.") # Added print

    # 1. Estandarizar nombres de columnas
    df_clean.columns = df_clean.columns.str.strip().str.upper()
    print("\n[1] Encabezados estandarizados a MAYÚSCULAS.")

    # 2. Parseo de fechas: 'on 02/02/17 19:53 PM (PST)' -> datetime
    if 'REVIEW_DATE' in df_clean.columns:
        print("[2] Parseando fechas...")
        # Clean date string: remove 'on ' prefix, timezone in parentheses, and AM/PM indicators
        # Example: 'on 04/28/17 08:08 AM (PDT)' -> '04/28/17 08:08 AM'
        df_clean['REVIEW_DATE_TEMP'] = df_clean['REVIEW_DATE'].astype(str)
        # Remove 'on ' at the beginning
        df_clean['REVIEW_DATE_TEMP'] = df_clean['REVIEW_DATE_TEMP'].apply(lambda x: re.sub(r'^on\s*', '', x))
        # Remove timezone like '(PDT)'
        df_clean['REVIEW_DATE_TEMP'] = df_clean['REVIEW_DATE_TEMP'].apply(lambda x: re.sub(r'\s+\(\w{3}\)$', '', x))
        # Remove AM/PM indicators (case-insensitive) to standardize to 24-hour format
        df_clean['REVIEW_DATE_TEMP'] = df_clean['REVIEW_DATE_TEMP'].apply(lambda x: re.sub(r'\s*(AM|PM)', '', x, flags=re.IGNORECASE).strip())

        df_clean['REVIEW_DATE_PARSED'] = pd.to_datetime(
            df_clean['REVIEW_DATE_TEMP'], format='%m/%d/%y %H:%M', errors='coerce' # Changed format to 24-hour without AM/PM
        )
        # Extraer año de la reseña
        df_clean['REVIEW_YEAR'] = df_clean['REVIEW_DATE_PARSED'].dt.year

        # Print range only if valid dates exist
        if not df_clean['REVIEW_YEAR'].isnull().all():
             print(f"   Fechas parseadas correctamente. Rango: {int(df_clean['REVIEW_YEAR'].min())} - {int(df_clean['REVIEW_YEAR'].max())}")
        else:
             print("   Advertencia: No se pudieron parsear fechas válidas.")

        df_clean = df_clean.drop(columns=['REVIEW_DATE_TEMP']) # Clean up intermediate column

    # 3. Extraer Año, Marca y Modelo desde VEHICLE_TITLE
    #    Formato típico: "2004 Dodge Neon SRT-4 SRT-4 4dr Sedan (2.4L 4cyl Turbo 5M)"
    if 'VEHICLE_TITLE' in df_clean.columns:
        print("[3] Extrayendo año, marca y modelo desde VEHICLE_TITLE...")
        # Extraer año (primeros 4 dígitos)
        df_clean['CAR_YEAR'] = df_clean['VEHICLE_TITLE'].str.extract(r'(\b\d{4}\b)')[0].fillna('No especificado') # Ensure it's a series and fillna
        # Marca estandarizada desde la columna Brand
        if 'BRAND' in df_clean.columns:
            df_clean['CAR_MAKE'] = df_clean['BRAND'].str.strip().str.title()
        # Modelo: extraer después del año y la marca
        df_clean['CAR_MODEL'] = df_clean['VEHICLE_TITLE'].apply(
            lambda x: ' '.join(str(x).split()[2:4]) if isinstance(x, str) and len(str(x).split()) > 3 else 'No especificado'
        )

    # 4. Limpieza de texto en reseñas
    if 'REVIEW' in df_clean.columns:
        print("[4] Limpiando texto de reseñas...")
        df_clean['REVIEW'] = df_clean['REVIEW'].astype(str).str.strip()
        # Remover HTML tags si existen
        df_clean['REVIEW'] = df_clean['REVIEW'].str.replace(r'<[^>]+>', '', regex=True)
        # Remover espacios múltiples
        df_clean['REVIEW'] = df_clean['REVIEW'].str.replace(r'\s+', ' ', regex=True)
    if 'REVIEW_TITLE' in df_clean.columns:
        df_clean['REVIEW_TITLE'] = df_clean['REVIEW_TITLE'].astype(str).str.strip()

    # 5. Manejo de nulos
    print("[5] Manejando valores nulos...")
    # Columnas de texto: reemplazar nulos con 'No especificado'
    cols_texto = ['AUTHOR_NAME', 'REVIEW_TITLE', 'CAR_MODEL', 'CAR_YEAR'] # Added CAR_YEAR
    for col in cols_texto:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('No especificado')
            df_clean[col] = df_clean[col].replace(
                ['nan', 'NaN', 'None', '', 'No especificado'], 'No especificado'
            )
    # Filtrar filas sin Review (no sirven para RAG)
    antes = len(df_clean)
    # Keep only rows where REVIEW is not empty or 'nan' string
    df_clean = df_clean[df_clean['REVIEW'].astype(str).str.strip() != '']
    df_clean = df_clean[df_clean['REVIEW'].astype(str) != 'nan']
    print(f"   Filas eliminadas por review vacía: {antes - len(df_clean)}")

    # 6. Estandarizar marcas a mayúsculas
    if 'CAR_MAKE' in df_clean.columns:
        df_clean['CAR_MAKE'] = df_clean['CAR_MAKE'].str.upper()

    print(f"\n[RESUMEN] Dataset normalizado: {len(df_clean)} filas")
    print(f"[RESUMEN] Nulos remanentes totales: {df_clean.isnull().sum().sum()}")

    return df_clean


if df is not None:
    print(f"[PRE-NORMALIZACION] df tiene {df.shape[0]} filas.") # Added print
    df_limpio = normalizar_dataset(df)

    # Guardar versión normalizada
    df_limpio.to_csv('autos_normalizado.csv', index=False)
    print("\n[SISTEMA] Dataset normalizado guardado como 'autos_normalizado.csv'.")
    display(df_limpio.head(2))

# ==========================================================
# ---
# # ==========================================
# # AGENTE 2B: ENTRENADOR
# # (Preparación, Entrenamiento y Selección de Modelos)
# # ==========================================
# 
# Este agente toma el dataset normalizado y:
# 1. **Prepara** features numéricas y textuales para ML clásico
# 2. **Entrena** múltiples modelos (Regresión Lineal, Random Forest, XGBoost)
# 3. **Evalúa** con validación cruzada y selecciona el mejor
# 4. **Guarda** el modelo ganador para inferencia
# 
# **Target**: `RATING` (1.0 - 5.0) → problema de **regresión**
# 
# **Features**:
# - `CAR_YEAR`: año del vehículo
# - `REVIEW_YEAR`: año de la reseña
# - `CAR_MAKE`: marca (codificada)
# - `REVIEW_LEN`: longitud del texto de la reseña
# - `TITLE_LEN`: longitud del título
# - Sentimiento calculado desde el texto
# ==========================================================

# ============================================================
# AGENTE 2B — FASE 1: INGENIERÍA DE CARACTERÍSTICAS Y SPLIT
# ============================================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

print("="*60)
print("    AGENTE 2B: INGENIERÍA DE CARACTERÍSTICAS")
print("="*60)

# Verificar que df_limpio existe
if 'df_limpio' not in dir() and 'df_limpio' not in globals():
    print("[ERROR] df_limpio no encontrado. Ejecuta primero el Agente 2 (Normalización).")
    # Fallback: intentar cargar desde CSV
    try:
        df_limpio = pd.read_csv('autos_normalizado.csv')
        print("[OK] Dataset cargado desde autos_normalizado.csv")
    except:
        raise RuntimeError("No hay datos disponibles. Ejecuta la normalización primero.")

df_train = df_limpio.copy()
print(f"[DATASET] {len(df_train)} filas disponibles")

# --- 1. Feature: longitud del texto ---
print("\n[1] Creando features de longitud textual...")
df_train['REVIEW_LEN'] = df_train['REVIEW'].astype(str).str.len()
df_train['TITLE_LEN'] = df_train['REVIEW_TITLE'].astype(str).str.len()

# --- 2. Feature: año del vehículo como numérico ---
print("[2] Procesando año del vehículo...")
df_train['CAR_YEAR_NUM'] = pd.to_numeric(df_train['CAR_YEAR'], errors='coerce')

# --- 3. Feature: año de la reseña ---
print("[3] Procesando año de la reseña...")
df_train['REVIEW_YEAR'] = pd.to_numeric(df_train['REVIEW_YEAR'], errors='coerce')

# --- 4. Codificar marca (CAR_MAKE) ---
print("[4] Codificando marca del vehículo...")
le_make = LabelEncoder()
df_train['MAKE_CODE'] = le_make.fit_transform(df_train['CAR_MAKE'].fillna('DESCONOCIDO'))
print(f"   Marcas únicas codificadas: {len(le_make.classes_)}")

# --- 5. Reducir a muestra representativa para ML ---
print("[5] Reduciendo a muestra representativa para ML...")
MUESTRA_ML = 20000
if len(df_train) > MUESTRA_ML:
    df_train = df_train.sample(n=MUESTRA_ML, random_state=42)
    print(f"   Dataset reducido a {len(df_train)} filas\n")
else:
    print(f"   Dataset tiene {len(df_train)} filas (no requiere reducción)\n")

# --- 6. TF-IDF de la reseña (top 100 términos) ---
print("[6] Vectorizando reseñas con TF-IDF (top 100 términos)...")
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_train['REVIEW'].fillna(''))
print(f"   Matriz TF-IDF: {tfidf_matrix.shape}")

# Convertir TF-IDF a DataFrame con prefijo
tfidf_cols = [f'TFIDF_{i}' for i in range(tfidf_matrix.shape[1])]
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df_train.index)

# --- 7. Feature: polaridad simple (cantidad de palabras positivas vs negativas) ---
print("[7] Calculando polaridad simple del texto...")
positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'awesome',
                  'fantastic', 'wonderful', 'reliable', 'comfortable', 'smooth', 'quick', 'fast',
                  'beautiful', 'fun', 'impressive', 'happy', 'satisfied'}
negative_words = {'bad', 'worst', 'terrible', 'horrible', 'poor', 'awful', 'hate', 'ugly',
                  'slow', 'unreliable', 'broken', 'expensive', 'disappointed', 'problem',
                  'issue', 'failure', 'regret', 'waste', 'trouble', 'defect'}

def simple_sentiment(text):
    words = set(str(text).lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0

df_train['SENTIMENT_SCORE'] = df_train['REVIEW'].apply(simple_sentiment)

# --- 8. Ensamblar matriz de features final ---
print("\n[8] Ensamblando matriz de features...")
feature_cols = ['REVIEW_LEN', 'TITLE_LEN', 'CAR_YEAR_NUM', 'REVIEW_YEAR', 'MAKE_CODE', 'SENTIMENT_SCORE']
X_numeric = df_train[feature_cols].fillna(0)

# Combinar numéricas con TF-IDF
X = np.hstack([X_numeric.values, tfidf_matrix.toarray()])
y = df_train['RATING'].values

print(f"   Features numéricas: {X_numeric.shape[1]}")
print(f"   Features TF-IDF: {tfidf_matrix.shape[1]}")
print(f"   Total features: {X.shape[1]}")
print(f"   Target (RATING) - min: {y.min():.1f}, max: {y.max():.1f}, media: {y.mean():.2f}")

# --- 9. Escalar features numéricas ---
print("\n[9] Escalando features numéricas (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 10. Train/Test Split ---
print("[10] Dividiendo en train (80%) y test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"   Train: {len(X_train)} muestras")
print(f"   Test:  {len(X_test)} muestras")

# --- 11. Guardar preprocesadores ---
print("\n[11] Guardando preprocesadores para producción...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le_make, 'label_encoder_make.pkl')
print("   [OK] scaler.pkl, tfidf_vectorizer.pkl, label_encoder_make.pkl")

print("\n" + "="*60)
print("    FASE 1 COMPLETADA — Datos listos para entrenamiento")
print("="*60)

# ==========================================================
# ### **FASE 2: ENTRENAMIENTO Y VALIDACIÓN DE MODELOS**
# 
# Entrenamos múltiples modelos de regresión y evaluamos con **validación cruzada 5-fold**:
# - **Regresión Lineal** (baseline)
# - **Ridge Regression** (regularización L2)
# - **Random Forest Regressor** (ensemble)
# - **XGBoost Regressor** (gradient boosting)
# 
# Métricas: **MAE**, **RMSE**, **R²**
# 
# ---
# ==========================================================

# ============================================================
# AGENTE 2B — FASE 2: ENTRENAMIENTO CON VALIDACIÓN CRUZADA
# ============================================================
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import time
warnings.filterwarnings('ignore')

print("="*60)
print("    AGENTE 2B — FASE 2: ENTRENAMIENTO DE MODELOS")
print("="*60)

# Verificar que X_train, y_train, X_test, y_test existen
if 'X_train' not in dir() and 'X_train' not in globals():
    print("[ERROR] Ejecuta primero la FASE 1 (feature engineering).")
    raise RuntimeError("FASE 1 no ejecutada.")

print(f"[TRAIN] {len(X_train)} muestras | [TEST] {len(X_test)} muestras\n")

# --- Usar una muestra representativa para acelerar entrenamiento ---
SAMPLE_SIZE = 15000  # Ajusta según RAM disponible
if len(X_train) > SAMPLE_SIZE:
    print(f"[INFO] Usando muestra de {SAMPLE_SIZE} filas para entrenamiento (dataset completo: {len(X_train)})")
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train), SAMPLE_SIZE, replace=False)
    X_sub = X_train[idx]
    y_sub = y_train[idx]
else:
    X_sub = X_train
    y_sub = y_train

# --- Diccionario de modelos ---
modelos = {
    'Regresión Lineal': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'XGBoost': None
}

# Intentar importar XGBoost (puede no venir preinstalado)
try:
    import xgboost as xgb
    modelos['XGBoost'] = xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
    print("[XGBoost] Importado correctamente.\n")
except ImportError:
    print("[XGBoost] No instalado. Se omitirá.\n")
    del modelos['XGBoost']

# --- Entrenar y evaluar cada modelo ---
resultados = []
kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold para velocidad

for nombre, modelo in modelos.items():
    print(f"{'─'*50}")
    print(f"  Modelo: {nombre}")
    print(f"{'─'*50}")
    t0 = time.time()

    # Validación cruzada (solo sobre la muestra)
    print("   Ejecutando validación cruzada 3-fold...")
    cv_r2 = cross_val_score(modelo, X_sub, y_sub, cv=kfold, scoring='r2')
    cv_mae = cross_val_score(modelo, X_sub, y_sub, cv=kfold, scoring='neg_mean_absolute_error')
    cv_rmse = cross_val_score(modelo, X_sub, y_sub, cv=kfold, scoring='neg_root_mean_squared_error')

    print(f"  CV R²:   {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")
    print(f"  CV MAE:  {-cv_mae.mean():.4f} (±{cv_mae.std():.4f})")
    print(f"  CV RMSE: {-cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")

    # Entrenar en la muestra y evaluar en test
    print("   Entrenando modelo final sobre la muestra...")
    modelo.fit(X_sub, y_sub)
    y_pred = modelo.predict(X_test)

    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = mean_squared_error(y_test, y_pred) ** 0.5

    elapsed = time.time() - t0
    print(f"  TEST R²:   {test_r2:.4f}")
    print(f"  TEST MAE:  {test_mae:.4f}")
    print(f"  TEST RMSE: {test_rmse:.4f}")
    print(f"  ⏱ Tiempo: {elapsed:.1f}s")

    resultados.append({
        'Modelo': nombre,
        'CV_R2_mean': cv_r2.mean(),
        'CV_R2_std': cv_r2.std(),
        'CV_MAE': -cv_mae.mean(),
        'CV_RMSE': -cv_rmse.mean(),
        'Test_R2': test_r2,
        'Test_MAE': test_mae,
        'Test_RMSE': test_rmse
    })
    print()

# --- Tabla comparativa ---
print('\n' + '='*60)
print('    TABLA COMPARATIVA DE MODELOS')
print('='*60)
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('Test_R2', ascending=False)
display(df_resultados.round(4))

# Identificar el mejor modelo
mejor = df_resultados.iloc[0]
print(f"\n  >> MEJOR MODELO: {mejor['Modelo']} (R² test: {mejor['Test_R2']:.4f})")

print("\n" + "="*60)
print("    FASE 2 COMPLETADA — Modelos entrenados y evaluados")
print("="*60)

# ==========================================================
# ### **FASE 3: OPTIMIZACIÓN, SELECCIÓN FINAL Y GUARDADO**
# 
# Tomamos el mejor modelo de la FASE 2 y:
# 1. **Optimizamos hiperparámetros** con GridSearchCV
# 2. **Entrenamos el modelo definitivo** con todos los datos de entrenamiento
# 3. **Guardamos** el modelo (.pkl) para producción
# 4. **Demo de inferencia** con un ejemplo real
# 
# ---
# ==========================================================

# ============================================================
# AGENTE 2B — FASE 3: OPTIMIZACIÓN Y MODELO FINAL
# ============================================================
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np

print("="*60)
print("    AGENTE 2B — FASE 3: OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("="*60)

if 'X_train' not in dir() and 'X_train' not in globals():
    print("[ERROR] Ejecuta primero las FASES 1 y 2.")
    raise RuntimeError("FASES 1-2 no ejecutadas.")

print(f"[TRAIN] {len(X_train)} muestras | [TEST] {len(X_test)} muestras\n")

# --- 1. GridSearchCV sobre RandomForest ---
print("[1] GridSearchCV: buscando mejores hiperparámetros para RandomForest...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_base, param_grid,
    cv=3, scoring='neg_mean_absolute_error',
    verbose=1, n_jobs=-1
)

# Usamos una muestra del 30% para que GridSearch sea rápido en Colab
sample_size = min(3000, len(X_train))
indices = np.random.RandomState(42).choice(len(X_train), sample_size, replace=False)
X_sample = X_train[indices]
y_sample = y_train[indices]
print(f"   Buscando en {sample_size} muestras (subset para velocidad)...")

grid_search.fit(X_sample, y_sample)

print(f"\n   Mejores parámetros: {grid_search.best_params_}")
print(f"   Mejor CV MAE: {-grid_search.best_score_:.4f}")

# --- 2. Entrenar modelo definitivo ---
print("\n[2] Entrenando modelo definitivo con TODOS los datos...")
best_params = grid_search.best_params_
modelo_final = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
modelo_final.fit(X_train, y_train)

# Evaluar en test
y_pred_final = modelo_final.predict(X_test)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
rmse = mean_squared_error(y_test, y_pred_final) ** 0.5

print(f"\n   RESULTADOS MODELO FINAL:")
print(f"   R²:   {r2:.4f}")
print(f"   MAE:  {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")

# --- 3. Guardar modelo ---
print("\n[3] Guardando modelo final...")
joblib.dump(modelo_final, 'modelo_rating_final.pkl')
print("   [OK] modelo_rating_final.pkl")

# Guardar también feature columns para referencia
feature_info = {
    'numeric_features': feature_cols,
    'n_tfidf': tfidf_matrix.shape[1],
    'total_features': X.shape[1],
    'best_params': best_params,
    'test_r2': r2,
    'test_mae': mae,
    'test_rmse': rmse
}
joblib.dump(feature_info, 'feature_info.pkl')
print("   [OK] feature_info.pkl")

print("\n" + "="*60)
print("    MODELO FINAL LISTO PARA PRODUCCIÓN")
print("="*60)
# ============================================================
# AGENTE 2B — DEMO: INFERENCIA CON NUEVOS DATOS
# ============================================================
import joblib
import pandas as pd

print("="*60)
print("    DEMO: PREDECIR RATING DE UNA RESEÑA NUEVA")
print("="*60)

# Cargar artefactos guardados
try:
    scaler = joblib.load('scaler.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le_make = joblib.load('label_encoder_make.pkl')
    modelo = joblib.load('modelo_rating_final.pkl')
    print("[OK] Todos los artefactos cargados correctamente.\n")
except FileNotFoundError as e:
    print(f"[ERROR] No se encuentra: {e}")
    print("Ejecuta las FASES 1-3 primero para generar los archivos.")
    exit()

# --- Ejemplo 1 ---
print("─"*50)
print("Ejemplo 1: Reseña positiva")
print("─"*50)
review_text = "This car is absolutely amazing! Great performance, comfortable ride, and excellent fuel economy. I love it!"
review_title = "Best car ever!"
car_make = "FERRARI"
car_year = "2020"
review_year = 2023

# Feature engineering
review_len = len(review_text)
title_len = len(review_title)
car_year_num = float(car_year)
make_code = le_make.transform([car_make.upper()])[0]

# Sentiment simple
positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'awesome',
                  'fantastic', 'wonderful', 'reliable', 'comfortable', 'smooth', 'quick', 'fast',
                  'beautiful', 'fun', 'impressive', 'happy', 'satisfied'}
negative_words = {'bad', 'worst', 'terrible', 'horrible', 'poor', 'awful', 'hate', 'ugly',
                  'slow', 'unreliable', 'broken', 'expensive', 'disappointed', 'problem',
                  'issue', 'failure', 'regret', 'waste', 'trouble', 'defect'}
words = set(review_text.lower().split())
pos = len(words & positive_words)
neg = len(words & negative_words)
total = pos + neg
sentiment = (pos - neg) / total if total > 0 else 0

# TF-IDF
tfidf_vec = tfidf.transform([review_text]).toarray()

# Numeric features
numeric = np.array([[review_len, title_len, car_year_num, review_year, make_code, sentiment]])
X_demo = np.hstack([numeric, tfidf_vec])
X_demo_scaled = scaler.transform(X_demo)

pred = modelo.predict(X_demo_scaled)[0]
print(f"   Review: \"{review_text[:60]}...\"")
print(f"   Rating predicho: {pred:.2f} / 5.00\n")

# --- Ejemplo 2 ---
print("─"*50)
print("Ejemplo 2: Reseña negativa")
print("─"*50)
review_text2 = "Terrible car, worst purchase ever. Poor quality, broken parts, slow and unreliable. I regret buying this."
review_title2 = "Horrible experience"
car_make2 = "FIAT"
car_year2 = "2015"
review_year2 = 2022

review_len2 = len(review_text2)
title_len2 = len(review_title2)
car_year_num2 = float(car_year2)
make_code2 = le_make.transform([car_make2.upper()])[0]

words2 = set(review_text2.lower().split())
pos2 = len(words2 & positive_words)
neg2 = len(words2 & negative_words)
total2 = pos2 + neg2
sentiment2 = (pos2 - neg2) / total2 if total2 > 0 else 0

tfidf_vec2 = tfidf.transform([review_text2]).toarray()
numeric2 = np.array([[review_len2, title_len2, car_year_num2, review_year2, make_code2, sentiment2]])
X_demo2 = np.hstack([numeric2, tfidf_vec2])
X_demo2_scaled = scaler.transform(X_demo2)

pred2 = modelo.predict(X_demo2_scaled)[0]
print(f"   Review: \"{review_text2[:60]}...\"")
print(f"   Rating predicho: {pred2:.2f} / 5.00")

print("\n" + "="*60)
print("    DEMO COMPLETADA — Modelo listo para inferencia")
print("="*60)

# ==========================================================
# ---
# # ==========================================
# # AGENTE 3: TRANSFORMERS Y SISTEMA RAG
# # ==========================================
# 
# Este agente implementa:
# 1. **Clasificación de sentimiento** con `transformers` (HuggingFace pipeline)
# 2. **Embeddings semánticos** con `sentence-transformers`
# 3. **Índice vectorial** con FAISS
# 4. **Cadena RAG** con Mistral Large 3
# ==========================================================


# ==========================================================
# ### **FASE 3.1: CONFIGURACIÓN DEL ENTORNO Y API MISTRAL**
# 
# Configuramos la API key de Mistral (usar Secretos en Colab) y conectamos con Mistral Large 3.
# ==========================================================

import os

# En Google Colab, usa la pestaña de Secretos (icono de llave) para configurar MISTRAL_API_KEY
# O bien descomenta y pega tu API key directamente (no recomendado para producción):
# os.environ["MISTRAL_API_KEY"] = "tu-api-key-aqui"

try:
    from google.colab import userdata
    api_key = userdata.get('MISTRAL_API_KEY')
    os.environ["MISTRAL_API_KEY"] = api_key
    print("[COLAB] API Key cargada desde Secretos.")
except ImportError:
    print("[ENTORNO] No detectado Google Colab. Usando variable de entorno existente.")
except Exception as e:
    print(f"[AVISO] No se pudo cargar API Key: {e}")

from langchain_mistralai import ChatMistralAI

def inicializar_mistral():
    """
    Inicializa el modelo Mistral Large 3 con temperatura 0
    para respuestas deterministas y precisas.
    """
    try:
        llm = ChatMistralAI(model="mistral-large-latest", temperature=0.0)
        test = llm.invoke("Responde únicamente: CONEXIÓN OK")
        print(f"[MISTRAL] {test.content.strip()}")
        return llm
    except Exception as e:
        print(f"[ERROR] No se pudo conectar con Mistral: {e}")
        print("[SOLUCIÓN] Configura MISTRAL_API_KEY en Secretos (Colab) o como variable de entorno.")
        return None

llm = inicializar_mistral()

# ==========================================================
# ### **FASE 3.2: ANÁLISIS DE SENTIMIENTO CON TRANSFORMERS**
# 
# Usamos el pipeline de `transformers` de HuggingFace para clasificar el sentimiento de cada reseña. Esto enriquece los documentos RAG con metadata emocional.
# ==========================================================

from transformers import pipeline
import numpy as np

# Cargar pipeline de sentiment analysis con DistilBERT (rápido en CPU)
# DistilBERT es una versión destilada de BERT que mantiene el 97% de precisión
# siendo 60% más rápida. Ideal para entornos con recursos limitados como Colab.
print("[TRANSFORMERS] Cargando pipeline de sentiment analysis (DistilBERT)...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)
print("[TRANSFORMERS] Pipeline listo.")


def analizar_sentimiento_lote(textos, batch_size=32):
    """
    Analiza el sentimiento de una lista de textos usando transformers.
    Retorna etiqueta (POSITIVE/NEGATIVE) y score de confianza.
    """
    resultados = sentiment_pipeline(textos, batch_size=batch_size)
    etiquetas = [r['label'] for r in resultados]
    scores = [r['score'] for r in resultados]
    return etiquetas, scores


# Tomamos una muestra representativa para el análisis de sentimiento
# (procesar 500K+ reseñas completas consumiría mucha memoria en Colab)
if df_limpio is not None:
    # Limitar a 1000 reseñas para mantener el notebook rápido en Colab
    muestra_size = min(1000, len(df_limpio))
    df_muestra = df_limpio.sample(n=muestra_size, random_state=42).copy()

    textos_review = df_muestra['REVIEW'].tolist()
    print(f"\n[TRANSFORMERS] Analizando sentimiento de {len(textos_review)} reseñas...")

    etiquetas, scores = analizar_sentimiento_lote(textos_review)
    df_muestra['SENTIMENT_LABEL'] = etiquetas
    df_muestra['SENTIMENT_SCORE'] = scores

    # Estadísticas de sentimiento
    print(f"\n[TRANSFORMERS] Distribución de sentimientos:")
    print(df_muestra['SENTIMENT_LABEL'].value_counts())
    print(f"\n[TRANSFORMERS] Score promedio POSITIVE: {df_muestra[df_muestra['SENTIMENT_LABEL'] == 'POSITIVE']['SENTIMENT_SCORE'].mean():.4f}")
    print(f"[TRANSFORMERS] Score promedio NEGATIVE: {df_muestra[df_muestra['SENTIMENT_LABEL'] == 'NEGATIVE']['SENTIMENT_SCORE'].mean():.4f}")

    # Mostrar ejemplos
    print("\n[EJEMPLOS] Reseñas con sentimiento POSITIVE (score alto):")
    display(df_muestra[df_muestra['SENTIMENT_LABEL'] == 'POSITIVE']\
            .sort_values('SENTIMENT_SCORE', ascending=False)[['REVIEW_TITLE', 'REVIEW', 'RATING', 'SENTIMENT_LABEL', 'SENTIMENT_SCORE']].head(2))
    print("\n[EJEMPLOS] Reseñas con sentimiento NEGATIVE (score alto):")
    display(df_muestra[df_muestra['SENTIMENT_LABEL'] == 'NEGATIVE']\
            .sort_values('SENTIMENT_SCORE', ascending=False)[['REVIEW_TITLE', 'REVIEW', 'RATING', 'SENTIMENT_LABEL', 'SENTIMENT_SCORE']].head(2))

# ==========================================================
# ### **FASE 3.3: CONSTRUCCIÓN DEL SISTEMA RAG**
# 
# Transformamos cada fila del dataset en un documento narrativo con metadata enriquecida (incluyendo sentimiento), generamos embeddings con `sentence-transformers`, indexamos con FAISS y ensamblamos la cadena RAG con Mistral.
# ==========================================================

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def construir_sistema_rag(df, llm):
    """
    Construye el pipeline RAG completo:
    1. Textualización semántica de cada fila
    2. Generación de embeddings con sentence-transformers
    3. Indexación FAISS
    4. Cadena RAG con LangChain Expression Language (LCEL)

    A diferencia de create_pandas_dataframe_agent que usa un prefix estático,
    aquí el contexto se recupera DINÁMICAMENTE desde FAISS, inyectándose
    automáticamente en la variable {context} del prompt.
    """
    documentos = []
    print("[RAG] Creando narrativas semánticas desde las reseñas...")

    for idx, fila in df.iterrows():
        # Construir narrativa con los datos más relevantes
        narrativa = (
            f"Reseña de auto - Marca: {fila.get('CAR_MAKE', 'N/E')}. "
            f"Modelo: {fila.get('CAR_MODEL', 'N/E')}. "
            f"Año del vehículo: {fila.get('CAR_YEAR', 'N/E')}. "
            f"Título de la reseña: {fila.get('REVIEW_TITLE', 'N/E')}. "
            f"Puntuación del consumidor: {fila.get('RATING', 'N/E')} de 5. "
            f"Sentimiento detectado: {fila.get('SENTIMENT_LABEL', 'N/E')} "
            f"(confianza: {fila.get('SENTIMENT_SCORE', 'N/E')}). "
            f"Texto completo de la reseña: {fila.get('REVIEW', 'N/E')}"
        )

        metadatos = {
            "marca": str(fila.get('CAR_MAKE', '')).upper(),
            "modelo": str(fila.get('CAR_MODEL', '')),
            "rating": float(fila.get('RATING', 0)) if pd.notna(fila.get('RATING')) else 0.0,
            "sentimiento": str(fila.get('SENTIMENT_LABEL', '')),
            "año_vehiculo": str(fila.get('CAR_YEAR', ''))
        }
        documentos.append(Document(page_content=narrativa, metadata=metadatos))

    # Generar embeddings usando Sentence Transformers (corre en CPU)
    print("[RAG] Generando embeddings con sentence-transformers/all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Construir índice FAISS en memoria
    print("[RAG] Indexando documentos en FAISS...")
    vector_db = FAISS.from_documents(documentos, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    print(f"[RAG] Índice FAISS creado con {len(documentos)} documentos.")

    # Prompt corporativo anti-alucinaciones
    system_prompt = (
        "Eres un Analista Senior de reseñas automotrices.\n"
        "Responde las preguntas del usuario basándote ESTRICTA y EXCLUSIVAMENTE "
        "en el contexto provisto abajo.\n"
        "Si el contexto no contiene los datos para responder, di textualmente: "
        "'Lo siento, la información disponible en el corpus analizado no contiene "
        "los datos específicos para responder esa pregunta.'\n\n"
        "CONTEXTO FACTUAL:\n{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Cadena RAG con LCEL moderna (reemplaza create_retrieval_chain)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Wrapper para mantener compatibilidad con invoke({"input": ...})
    class RAGWrapper:
        def __init__(self, chain):
            self.chain = chain
        def invoke(self, inputs):
            res = self.chain.invoke(inputs["input"])
            return {"answer": res}

    return RAGWrapper(rag_chain)


# Construir el sistema RAG usando la muestra con sentimiento
if 'df_muestra' in locals() and llm is not None:
    pipeline_rag = construir_sistema_rag(df_muestra, llm)
    print("\n[ÉXITO] Sistema RAG construido y listo para consultas.")

    # Prueba de control
    consulta_prueba = "¿Qué reseñas existen para autos Ferrari con rating alto?"
    resultado = pipeline_rag.invoke({"input": consulta_prueba})
    print(f"\n[TEST RAG] Consulta: '{consulta_prueba}'")
    print(f"[TEST RAG] Respuesta:\n{resultado['answer']}")
else:
    print("[ERROR] Datos o LLM no disponibles. Ejecuta las celdas anteriores primero.")

# ==========================================================
# ---
# # ==========================================
# # AGENTE 3B: COMUNICADOR
# # (Generación de Reporte en Lenguaje Natural)
# # ==========================================
# 
# Este agente toma todos los resultados de los agentes anteriores (datos normalizados, análisis de sentimiento, modelo ML, sistema RAG) y genera un **reporte ejecutivo completo en lenguaje natural** utilizando Mistral Large 3.
# 
# El reporte incluye:
# 1. Resumen del dataset analizado
# 2. Distribución de sentimientos encontrados
# 3. Rendimiento del modelo de predicción de rating
# 4. Top marcas mejor y peor valoradas
# 5. Conclusiones y recomendaciones basadas en los datos
# 
# ---
# ==========================================================

# ============================================================
# AGENTE 3B: COMUNICADOR — GENERACIÓN DE REPORTE NATURAL
# ============================================================
import json
import time
import pandas as pd
import numpy as np

print("="*60)
print("    AGENTE COMUNICADOR: GENERANDO REPORTE EJECUTIVO")
print("="*60)

# --- 1. Recolectar metadata del dataset ---
print("\n[1] Recolectando metadata del análisis...")

total_filas = len(df_limpio) if 'df_limpio' in dir() or 'df_limpio' in globals() else 0
total_marcas = df_limpio['CAR_MAKE'].nunique() if total_filas > 0 else 0

# Distribución de sentimiento (si existe df_muestra)
if 'df_muestra' in dir() or 'df_muestra' in globals():
    sent_dist = df_muestra['SENTIMENT_LABEL'].value_counts().to_dict()
    pos_pct = (sent_dist.get('POSITIVE', 0) / len(df_muestra)) * 100
    neg_pct = (sent_dist.get('NEGATIVE', 0) / len(df_muestra)) * 100
    sent_score_pos = df_muestra[df_muestra['SENTIMENT_LABEL'] == 'POSITIVE']['SENTIMENT_SCORE'].mean()
    sent_score_neg = df_muestra[df_muestra['SENTIMENT_LABEL'] == 'NEGATIVE']['SENTIMENT_SCORE'].mean()
else:
    sent_dist = {}
    pos_pct = neg_pct = 0
    sent_score_pos = sent_score_neg = 0

# Top marcas por rating promedio
if total_filas > 0:
    top_marcas = df_limpio.groupby('CAR_MAKE')['RATING'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    top_5 = top_marcas.head(5)
    bottom_5 = top_marcas.tail(5)
    rating_global = df_limpio['RATING'].mean()
else:
    top_5 = bottom_5 = pd.DataFrame()
    rating_global = 0

# Distribución de ratings
if total_filas > 0:
    rating_counts = df_limpio['RATING'].value_counts().sort_index().to_dict()
else:
    rating_counts = {}

# --- 2. Recolectar métricas del modelo ML ---
print("[2] Recolectando métricas del modelo ML...")
try:
    feature_info = joblib.load('feature_info.pkl')
    mejor_r2 = feature_info.get('test_r2', 'N/A')
    mejor_mae = feature_info.get('test_mae', 'N/A')
    mejor_rmse = feature_info.get('test_rmse', 'N/A')
    mejor_params = feature_info.get('best_params', 'N/A')
    print("   [OK] Métricas cargadas desde feature_info.pkl")
except Exception as e:
    print(f"   [AVISO] No se encontraron métricas: {e}")
    mejor_r2 = mejor_mae = mejor_rmse = 'N/A'
    mejor_params = 'N/A'

# --- 3. Construir el reporte en lenguaje natural ---
print("[3] Generando reporte narrativo con Mistral...\n")

prompt_reporte = f"""
Eres un Analista Senior de Datos Automotrices. Genera un reporte ejecutivo PROFESIONAL y BIEN ESTRUCTURADO
en español basándote ESTRICTAMENTE en los siguientes datos reales del análisis.
Usa un tono formal, profesional y de consultoría.

--- DATOS DEL ANÁLISIS ---

1. DATASET:
   - Total de reseñas analizadas: {total_filas}
   - Marcas únicas: {total_marcas}
   - Rating global promedio: {rating_global:.2f} / 5.0
   - Distribución de ratings (estrellas): {json.dumps(rating_counts)}

2. ANÁLISIS DE SENTIMIENTO (DistilBERT):
   - Reseñas POSITIVAS: {sent_dist.get('POSITIVE', 0)} ({pos_pct:.1f}%)
   - Reseñas NEGATIVAS: {sent_dist.get('NEGATIVE', 0)} ({neg_pct:.1f}%)
   - Confianza promedio POSITIVE: {sent_score_pos:.4f}
   - Confianza promedio NEGATIVE: {sent_score_neg:.4f}

3. MODELO DE PREDICCIÓN (Random Forest):
   - R² en test: {mejor_r2}
   - MAE en test: {mejor_mae}
   - RMSE en test: {mejor_rmse}
   - Mejores hiperparámetros: {mejor_params}

4. TOP 5 MARCAS MEJOR VALORADAS:
{top_5.to_string()}

5. TOP 5 MARCAS PEOR VALORADAS:
{bottom_5.to_string()}

--- FORMATO DEL REPORTE ---

Genera un reporte que incluya las siguientes secciones:

## RESUMEN EJECUTIVO
(2-3 párrafos con los hallazgos más importantes)

## 1. PANORAMA DEL DATASET
(Descripción del volumen de datos, marcas cubiertas y distribución general de ratings)

## 2. ANÁLISIS DE SENTIMIENTO
(Interpretación de la distribución POSITIVE/NEGATIVE y la confianza del modelo)

## 3. RENDIMIENTO DEL MODELO PREDICTIVO
(Qué tan bien predice el rating, qué significan las métricas,限aciones)

## 4. RANKING DE MARCAS
(Análisis de las mejores y peores marcas según los consumidores)

## 5. CONCLUSIONES Y RECOMENDACIONES
(Recomendaciones accionables basadas en los datos)

IMPORTANTE: No inventes datos. Si algo no está disponible, indícalo.
Usa formato markdown limpio para el reporte.
"""

if llm is not None:
    try:
        respuesta = llm.invoke(prompt_reporte)
        reporte = respuesta.content

        print("="*60)
        print("            REPORTE EJECUTIVO GENERADO")
        print("="*60)
        print(reporte)
        print("="*60)

        # Guardar a archivo
        with open('reporte_ejecutivo_autos.md', 'w', encoding='utf-8') as f:
            f.write(reporte)
        print("\n[GUARDADO] Reporte guardado como 'reporte_ejecutivo_autos.md'")

    except Exception as e:
        print(f"[ERROR] No se pudo generar el reporte: {e}")
else:
    print("[ERROR] Mistral no está configurado. Ejecuta la Fase 3.1 primero.")

print("\n" + "="*60)
print("    AGENTE COMUNICADOR FINALIZADO")
print("="*60)

# ==========================================================
# ---
# # ==========================================
# # AGENTE 4: AGENTE AUTÓNOMO CON FUNCTION CALLING
# # ==========================================
# 
# Implementamos un agente autónomo que usa Mistral Large 3 como motor de razonamiento. El agente puede:
# - **Buscar en el RAG**: recuperar reseñas específicas por marca, modelo, sentimiento, etc.
# - **Calcular estadísticas**: promedios de rating, conteos, operaciones matemáticas
# 
# El agente decide AUTÓNOMAMENTE qué herramienta usar según la pregunta del usuario (bucle ReAct).
# ==========================================================

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, ToolMessage
import pandas as pd
import time

# =====================================================================
# HERRAMIENTA 1: Buscador RAG de reseñas de autos
# =====================================================================
def tool_buscar_resenas(query):
    """
    Busca en el índice vectorial FAISS las reseñas más relevantes
    según la consulta semántica del usuario.
    """
    if 'pipeline_rag' not in locals() and 'pipeline_rag' not in globals():
        return "Error: El sistema RAG no está inicializado."
    res = pipeline_rag.invoke({"input": query})
    return res["answer"]

# =====================================================================
# HERRAMIENTA 2: Calculadora/Python para análisis numérico
# =====================================================================
def tool_calculadora(expresion):
    """
    Evalúa expresiones matemáticas de forma segura.
    Útil para sumar ratings, promediar precios, etc.
    NOTA: Solo permite operaciones básicas (suma, resta, multiplicación, división).
    """
    try:
        # Entorno restringido por seguridad (no permite builtins peligrosos)
        res = eval(str(expresion), {"__builtins__": {}}, {})
        return f"Resultado del cálculo: {res}"
    except Exception as e:
        return f"No se pudo calcular. Error: {e}"

# =====================================================================
# HERRAMIENTA 3: Estadísticas del DataFrame
# =====================================================================
def tool_estadisticas_dataframe(consulta):
    """
    Ejecuta consultas de análisis sobre el DataFrame completo.
    Consultas soportadas: 'promedio rating {marca}', 'conteo {marca}', 'top marcas'
    """
    if 'df_limpio' not in dir() and 'df_limpio' not in globals():
        return "Error: DataFrame no disponible."

    consulta = str(consulta).lower()

    if 'promedio rating' in consulta:
        for marca in df_limpio['CAR_MAKE'].unique():
            if marca.lower() in consulta:
                prom = df_limpio[df_limpio['CAR_MAKE'] == marca]['RATING'].mean()
                count = df_limpio[df_limpio['CAR_MAKE'] == marca]['RATING'].count()
                return f"Rating promedio para {marca}: {prom:.2f} (basado en {count} reseñas)"
        return "Marca no encontrada. Especifica una marca válida."

    if 'conteo' in consulta or 'count' in consulta or 'cuántas' in consulta:
        for marca in df_limpio['CAR_MAKE'].unique():
            if marca.lower() in consulta:
                count = len(df_limpio[df_limpio['CAR_MAKE'] == marca])
                return f"Cantidad de reseñas para {marca}: {count}"
        return "Marca no encontrada."

    if 'top' in consulta or 'mejores' in consulta:
        top = df_limpio.groupby('CAR_MAKE')['RATING'].mean().sort_values(ascending=False).head(5)
        return f"Top 5 marcas mejor rating:\n{top.to_string()}"

    return "Consulta no reconocida. Prueba con: 'promedio rating Porsche', 'conteo Ferrari', 'top marcas'"


# Registrar las herramientas disponibles para el agente
herramientas = [
    Tool(
        name="Buscador_Resenas_RAG",
        func=tool_buscar_resenas,
        description="Busca reseñas de autos en el índice semántico. Útil para preguntas sobre reseñas específicas, "
                    "experiencias de usuarios, opiniones detalladas. Recibe una consulta en lenguaje natural."
    ),
    Tool(
        name="Calculadora",
        func=tool_calculadora,
        description="Realiza cálculos matemáticos exactos. Recibe una expresión como '4.5 + 3.2 + 5.0 / 3'. "
                    "Útil para promediar ratings, sumar valores, etc."
    ),
    Tool(
        name="Estadisticas_DataFrame",
        func=tool_estadisticas_dataframe,
        description="Obtiene estadísticas del dataset: promedio de rating por marca, conteo de reseñas, top marcas. "
                    "Recibe comandos como 'promedio rating Ferrari', 'conteo Porsche', 'top marcas'."
    )
]

print("[AGENTE] Herramientas registradas:")
for h in herramientas:
    print(f"  -> {h.name}: {h.description[:60]}...")
def ejecutar_agente(pregunta, llm, tools):
    """
    Bucle principal del agente autónomo.

    Funcionamiento:
    1. El LLM recibe la pregunta del usuario
    2. Decide si necesita usar una herramienta (tool_call)
    3. Si llama a una herramienta, ejecuta la función y devuelve el resultado
    4. El LLM revisa el resultado y decide si necesita más herramientas
    5. Cuando tiene suficiente información, genera la respuesta final

    Límite de 5 iteraciones para evitar loops infinitos.
    """
    print("="*60)
    print("        INICIALIZANDO AGENTE AUTÓNOMO")
    print("="*60)

    dict_tools = {t.name: t.func for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    mensajes = [
        HumanMessage(content=(
            "Eres un Agente Autónomo de Análisis Automotriz.\n"
            "Tu misión es responder preguntas sobre reseñas de autos usando las herramientas disponibles.\n"
            "Sé preciso, profesional y fundamenta tus respuestas en los datos.\n"
            "Si necesitas calcular promedios, usa la Calculadora.\n"
            "Si necesitas buscar reseñas, usa el Buscador_RAG.\n"
            "Si necesitas estadísticas del dataset, usa Estadisticas_DataFrame.\n\n"
            f"PREGUNTA DEL USUARIO: {pregunta}"
        ))
    ]

    for paso in range(5):
        # Pausa para evitar rate limiting (Error 429)
        time.sleep(10) # Aumentado el tiempo de espera para el LLM y herramientas a 10 segundos

        print(f"\n[PASO {paso + 1}] El agente está analizando...")
        respuesta = llm_with_tools.invoke(mensajes)
        mensajes.append(respuesta)

        if hasattr(respuesta, 'tool_calls') and respuesta.tool_calls:
            for tool_call in respuesta.tool_calls:
                nombre = tool_call['name']
                args = tool_call['args']
                tool_id = tool_call['id']

                print(f"  -> [HERRAMIENTA] {nombre}")

                if isinstance(args, dict) and args:
                    parametro = list(args.values())[0]
                else:
                    parametro = str(args)

                print(f"  -> [INPUT] {parametro[:100]}...")

                if nombre in dict_tools:
                    observacion = dict_tools[nombre](parametro)
                else:
                    observacion = f"Error: Herramienta '{nombre}' no disponible."

                print(f"  -> [OBSERVACIÓN] {str(observacion)[:150]}...")
                mensajes.append(ToolMessage(content=str(observacion), tool_call_id=tool_id))
        else:
            print("\n" + "="*60)
            print("           RESPUESTA FINAL DEL AGENTE")
            print("="*60)
            print(respuesta.content)
            print("="*60)
            return

    print("\n[LÍMITE] El agente alcanzó el máximo de pasos.")


# =====================================================================
# EJECUCIÓN DEL AGENTE
# =====================================================================
if llm is not None:
    if 'pipeline_rag' in locals() or 'pipeline_rag' in globals():
        print("\n" + "#"*60)
        print("#       EJEMPLO 1: CONSULTA COMBINADA (RAG + ESTADÍSTICAS)")
        print("#"*60)
        pregunta1 = "Dame un resumen de las reseñas de Porsche y calcula el rating promedio de la marca"
        ejecutar_agente(pregunta1, llm, herramientas)

        time.sleep(30) # Añadido un sleep adicional y aumentado el tiempo entre ejecuciones a 30 segundos

        print("\n" + "#"*60)
        print("#       EJEMPLO 2: CONSULTA COMPARATIVA")
        print("#"*60)
        pregunta2 = "¿Cuántas reseñas hay de Ferrari y cuál es su rating promedio? Compáralo con Lamborghini"
        ejecutar_agente(pregunta2, llm, herramientas)
    else:
        print("[ERROR] El pipeline RAG no está inicializado. Ejecuta la Fase 3.3 primero.")
else:
    print("[ERROR] Mistral no está configurado. Ejecuta la Fase 3.1 primero.")

# ==========================================================
# ---
# # **CONCLUSIÓN**
# 
# ## **Resumen de lo implementado**
# 
# | Componente | Tecnología | Propósito |
# |------------|-----------|-----------|
# | **Agente 1** | Análisis de dataset | Definir y justificar la elección del dataset |
# | **Agente 2** | Pandas + EDA | Carga, limpieza y normalización de datos |
# | **Agente 2B** | Scikit-learn + XGBoost | Entrenamiento y optimización de modelo predictivo de rating |
# | **Agente 3 - Transformers** | DistilBERT (HuggingFace) | Análisis de sentimiento de reseñas |
# | **Agente 3 - RAG** | sentence-transformers + FAISS + Mistral | Búsqueda semántica y generación aumentada |
# | **Agente 3B - Comunicador** | Mistral Large 3 | Generación de reporte ejecutivo en lenguaje natural |
# | **Agente 4** | LangChain + Function Calling | Agente autónomo con razonamiento y herramientas |
# 
# ## **Próximos pasos**
# - Aumentar la muestra de reseñas para mejorar la cobertura del RAG
# - Probar con otros modelos de transformers (RoBERTa, DeBERTa)
# - Agregar más herramientas al agente (gráficos, clustering, etc.)
# - Desplegar como aplicación web con Gradio o Streamlit
# 
# ---
# ==========================================================

