import os
import requests
import pandas as pd
import pyarrow as pa
import s3fs
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from deltalake.writer import write_deltalake
from deltalake import DeltaTable

#CONFIGURACIONES Y CONSTANTES GLOBALES
MINIO_ENDPOINT = "http://minio:9000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")

STORAGE_OPTIONS_DELTA = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true"
}

# Rutas de almacenamiento
BRONZE_PATH = "s3://bronze/meteo_raw_delta"

def extract_and_load_bronze():
    print("1. Descargando datos crudos desde Open-Meteo API...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 41.3874,
        "longitude": 2.1686,
        "start_date": "2022-01-01",
        "end_date": "2025-12-31",
        "hourly": [
            "temperature_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "weather_code"
        ],
        "timezone": "Europe/Madrid"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df_meteo = pd.DataFrame(data["hourly"])

        df_meteo['etl_fecha_ingesta'] = pd.Timestamp.now(tz='UTC')

        print(f"Datos descargados. Filas totales: {len(df_meteo)}")

        fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
        if not fs.exists("bronze"):
            fs.mkdir("bronze")

        tabla_arrow = pa.Table.from_pandas(df_meteo)
        write_deltalake(
            table_or_uri=BRONZE_PATH,
            data=tabla_arrow,
            mode="overwrite",
            schema_mode="overwrite",
            storage_options=STORAGE_OPTIONS_DELTA
        )
        print(f"Carga cruda finalizada exitosamente en: {BRONZE_PATH}")

    except Exception as e:
        print(f"Error en la extracción a Bronze: {e}")
        raise e

# DEFINICIÓN DEL DAG
with DAG(
        dag_id='ingesta_meteo_bronze',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['meteo', 'bronze', 'silver', 'delta'],
) as dag:
    task_bronze = PythonOperator(
        task_id='extraer_meteo_bronze',
        python_callable=extract_and_load_bronze,
    )
    trigger_silver = TriggerDagRunOperator(
        task_id="weather_barna",
        trigger_dag_id="limpieza_meteo_silver",
        conf={"bronze_file_path": "{{ ti.xcom_pull(task_ids='fetch_api_to_bronze') }}"}
    )

    task_bronze >> trigger_silver