import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pyarrow as pa
import s3fs
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
# CONFIGURACIONES GLOBALES
MINIO_ENDPOINT = "http://minio:9000"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")

STORAGE_OPTIONS_DELTA = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true"
}

# Credenciales para que MLflow pueda descargar el modelo
os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT


def ejecutar_predicciones_2026():
    # CARGA DE DATOS
    print("Cargando escenarios base de 2026...")
    dt_sim = DeltaTable("s3://silver/accidentes_simulados_2026_delta", storage_options=STORAGE_OPTIONS_DELTA)
    df_2026 = dt_sim.to_pandas()
    df_2026.columns = df_2026.columns.str.lower()
    df_2026 = df_2026.rename(columns={'mes_any': 'mes'})

    #DATOS METEO Y TRÁFICO
    print("Calculando promedios históricos de Meteo y Tráfico...")

    dt_meteo = DeltaTable("s3://silver/meteo_delta", storage_options=STORAGE_OPTIONS_DELTA)
    df_meteo = dt_meteo.to_pandas()
    df_meteo.columns = df_meteo.columns.str.lower()
    df_meteo['data'] = pd.to_datetime(df_meteo['data'])
    df_meteo['mes'] = df_meteo['data'].dt.month

    meteo_avg = df_meteo.groupby(['mes', 'descripcio_torn']).agg({
        'temperature_2m': 'mean',
        'precipitation': 'mean',
        'rain': 'mean',
        'wind_speed_10m': 'mean',
        'weather_code': 'max'
    }).reset_index()

    dt_trafico = DeltaTable("s3://silver/trafico_delta", storage_options=STORAGE_OPTIONS_DELTA)
    df_trafico = dt_trafico.to_pandas()
    df_trafico.columns = df_trafico.columns.str.lower()
    df_trafico['data'] = pd.to_datetime(df_trafico['data'])
    df_trafico['mes'] = df_trafico['data'].dt.month

    trafico_avg = df_trafico.groupby(['codi_districte', 'mes', 'descripcio_torn']).agg({
        'pct_congestion': 'mean'
    }).reset_index()

    df_2026['mes'] = df_2026['mes'].astype(int)
    df_2026['codi_districte'] = df_2026['codi_districte'].astype(int)

    meteo_avg['mes'] = meteo_avg['mes'].astype(int)

    trafico_avg['mes'] = trafico_avg['mes'].astype(int)
    trafico_avg['codi_districte'] = trafico_avg['codi_districte'].astype(int)

    print("Cruzando variables...")
    df_2026 = df_2026.merge(meteo_avg, on=['mes', 'descripcio_torn'], how='left')
    df_2026 = df_2026.merge(trafico_avg, on=['codi_districte', 'mes', 'descripcio_torn'], how='left')

    # Limpieza de nulos
    df_2026['pct_congestion'] = df_2026['pct_congestion'].fillna(df_2026['pct_congestion'].mean())
    df_2026 = df_2026.fillna(0)

    #PREPARACIÓN DE FEATURES
    print("Codificando variables categóricas...")
    le_turno = LabelEncoder().fit(['Matí', 'Nit', 'Tarda'])
    le_dia = LabelEncoder().fit(['Dilluns', 'Dimarts', 'Dimecres', 'Dijous', 'Divendres', 'Dissabte', 'Diumenge'])

    df_2026['turno_enc'] = le_turno.transform(df_2026['descripcio_torn'].str.capitalize())
    df_2026['dia_enc'] = le_dia.transform(df_2026['descripcio_dia_setmana'].str.capitalize())

    #PREDICCIÓN CON MLFLOW
    print("Descargando modelo de MLflow y Prediciendo...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    modelo = mlflow.sklearn.load_model("models:/RiesgoAccidentesRF/latest")

    FEATURES = [
        'codi_districte', 'turno_enc', 'dia_enc', 'mes', 'dia_mes',
        'temperature_2m', 'precipitation', 'rain', 'wind_speed_10m',
        'weather_code', 'pct_congestion'
    ]

    X = df_2026[FEATURES]

    df_2026['prediccion_riesgo_enc'] = modelo.predict(X)
    df_2026['probabilidad_maxima'] = np.max(modelo.predict_proba(X), axis=1)

    #GUARDAR EN LA CAPA GOLD
    print("Exportando resultados a la capa GOLD...")
    TARGET_BUCKET = "gold"
    delta_table_path = f"s3://{TARGET_BUCKET}/predicciones_2026_delta"

    fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
    if not fs.exists(TARGET_BUCKET):
        fs.mkdir(TARGET_BUCKET)

    df_2026 = df_2026.astype({"data": "str"})
    tabla_arrow = pa.Table.from_pandas(df_2026)

    write_deltalake(
        table_or_uri=delta_table_path,
        data=tabla_arrow,
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=STORAGE_OPTIONS_DELTA
    )

    print(f"Las predicciones están listas en {delta_table_path}")

# DEFINICIÓN DEL DAG
with DAG(
        dag_id='prediccion_accidentes_2026',
        schedule='@daily',
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['inferencia', 'gold', 'mlflow', 'delta'],
) as dag:
    tarea_prediccion = PythonOperator(
        task_id='ejecutar_inferencia_2026',
        python_callable=ejecutar_predicciones_2026,
    )