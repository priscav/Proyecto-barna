import os
import pandas as pd
import pyarrow as pa
import s3fs
from datetime import datetime
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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


def limpiar_datos_silver():
    print("1. Leyendo datos RAW desde BRONZE...")
    dt_bronze = DeltaTable("s3://bronze/predicciones_raw_2026_delta", storage_options=STORAGE_OPTIONS)
    df_clean = dt_bronze.to_pandas()

    print("2. Aplicando reglas de limpieza...")
    # Eliminar columnas de encoding generadas solo para el algoritmo
    columnas_a_borrar = ['turno_enc', 'dia_enc']
    df_clean = df_clean.drop(columns=[col for col in columnas_a_borrar if col in df_clean.columns])

    # Redondeo de decimales infinitos
    df_clean['probabilidad_alto_riesgo'] = df_clean['probabilidad_alto_riesgo'].round(3)
    df_clean['pct_congestion'] = df_clean['pct_congestion'].round(3)

    # Estandarizar tipos
    df_clean['prediccion_riesgo_binario'] = df_clean['prediccion_riesgo_binario'].astype(int)

    print("3. Guardando datos limpios en SILVER...")
    write_deltalake(
        table_or_uri="s3://silver/predicciones_clean_2026_delta",
        data=pa.Table.from_pandas(df_clean),
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=STORAGE_OPTIONS
    )


with DAG(
        dag_id='limpieza_silver_prediccion_2026',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['silver', 'etl'],
) as dag:
    task_silver = PythonOperator(
        task_id='limpiar_datos',
        python_callable=limpiar_datos_silver,
    )

    trigger_gold = TriggerDagRunOperator(
        task_id="trigger_agregacion_gold",
        trigger_dag_id="prediccion_gold_2026",
        wait_for_completion=False,
    )

    task_silver >> trigger_gold