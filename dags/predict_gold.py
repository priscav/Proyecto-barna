import os
import pandas as pd
import pyarrow as pa
import s3fs
from datetime import datetime
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from airflow import DAG
from airflow.operators.python import PythonOperator

MINIO_ENDPOINT = "http://minio:9000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")

STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true"
}


def preparar_datos_gold():
    print("1. Leyendo datos limpios desde SILVER...")
    dt_silver = DeltaTable("s3://silver/predicciones_clean_2026_delta", storage_options=STORAGE_OPTIONS)
    df_gold = dt_silver.to_pandas()

    print("2. Seleccionando y agrupando información para negocio...")
    columnas_powerbi = [
        'data', 'codi_districte', 'nom_districte', 'dia_semana', 'descripcio_torn',
        'mes', 'temperature_2m', 'precipitation', 'pct_congestion',
        'prediccion_riesgo_binario', 'probabilidad_alto_riesgo'
    ]

    df_gold = df_gold[columnas_powerbi]

    print("3. Guardando tabla final en GOLD...")
    fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
    if not fs.exists("gold"):
        fs.mkdir("gold")

    write_deltalake(
        table_or_uri="s3://gold/predicciones_business_2026_delta",
        data=pa.Table.from_pandas(df_gold),
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=STORAGE_OPTIONS
    )
    print("Pipeline completado. Datos listos para Power BI en S3.")


with DAG(
        dag_id='prediccion_gold_2026',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['gold', 'powerbi'],
) as dag:
    task_gold = PythonOperator(
        task_id='preparar_vistas_negocio',
        python_callable=preparar_datos_gold,
    )