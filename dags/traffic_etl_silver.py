import os
import pandas as pd
import pyarrow as pa
import s3fs
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from deltalake.writer import write_deltalake
from deltalake import DeltaTable

# CONFIGURACIONES Y CONSTANTES GLOBALES
MINIO_ENDPOINT = "http://minio:9000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")

STORAGE_OPTIONS_DELTA = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true"
}

BRONZE_PATH = "s3://bronze/trafico_raw_delta"
SILVER_PATH = "s3://silver/trafico_delta"


def clean_and_load_silver():
    print(f"1. Leyendo datos crudos desde: {BRONZE_PATH}...")

    try:
        dt = DeltaTable(BRONZE_PATH, storage_options=STORAGE_OPTIONS_DELTA)
        df_trafico = dt.to_pandas()
    except Exception as e:
        raise ValueError(f"Error leyendo la tabla en Bronze. Detalles: {e}")

    print("Iniciando transformaciones para la capa Silver...")

    # Estandarizar a minusculas
    df_trafico.columns = df_trafico.columns.str.lower()

    #Formatear la fecha y tipos
    if 'fecha' in df_trafico.columns:
        df_trafico['data'] = pd.to_datetime(df_trafico['fecha']).dt.date
        df_trafico = df_trafico.drop(columns=['fecha'])

    if 'pct_congestion' in df_trafico.columns:
        df_trafico['pct_congestion'] = pd.to_numeric(df_trafico['pct_congestion'], errors='coerce')

    if 'codi_districte' in df_trafico.columns:
        df_trafico['codi_districte'] = pd.to_numeric(df_trafico['codi_districte'], errors='coerce')

    #Metadatos
    df_trafico['etl_fecha_procesamiento'] = pd.Timestamp.now(tz='UTC')

    print(f"   Transformación completada. Filas listas: {len(df_trafico)}")

    print("3. Guardando en capa Silver (MinIO)...")
    try:
        fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
        if not fs.exists("silver"):
            fs.mkdir("silver")

        tabla_arrow = pa.Table.from_pandas(df_trafico)

        write_deltalake(
            table_or_uri=SILVER_PATH,
            data=tabla_arrow,
            mode="overwrite",
            schema_mode="overwrite",
            storage_options=STORAGE_OPTIONS_DELTA
        )
        print(f"Carga agregada finalizada exitosamente en: {SILVER_PATH}")

    except Exception as e:
        print(f"Error guardando en Silver: {e}")
        raise e

# DEFINICIÓN DEL DAG
with DAG(
        dag_id='limpieza_trafico_silver',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['trafico', 'silver', 'delta'],
) as dag:
    tarea_silver = PythonOperator(
        task_id='transformar_trafico_silver',
        python_callable=clean_and_load_silver,
    )