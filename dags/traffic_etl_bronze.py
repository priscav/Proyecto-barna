import os
import pandas as pd
import pyarrow as pa
import s3fs
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from deltalake.writer import write_deltalake

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


def extract_and_load_bronze():
    FILE_PATH = "/opt/airflow/dags/files/trafico_agregado_distrito.csv"

    print(f"1. Leyendo archivo local desde: {FILE_PATH}")

    try:
        df_trafico = pd.read_csv(FILE_PATH)
        df_trafico['etl_fecha_ingesta'] = pd.Timestamp.now(tz='UTC')
        print(f"   Datos cargados. Filas totales: {len(df_trafico)}")

        # Aseguramos el bucket
        fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
        if not fs.exists("bronze"):
            fs.mkdir("bronze")

        # Convertir a string preventivamente para evitar conflictos de tipos en Bronze crudo
        df_trafico = df_trafico.astype(str)
        tabla_arrow = pa.Table.from_pandas(df_trafico)

        print("2. Guardando en capa Bronze (MinIO)...")
        write_deltalake(
            table_or_uri=BRONZE_PATH,
            data=tabla_arrow,
            mode="overwrite",
            schema_mode="overwrite",
            storage_options=STORAGE_OPTIONS_DELTA
        )
        print(f"Carga cruda finalizada exitosamente en: {BRONZE_PATH}")

    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el archivo en {FILE_PATH}. Asegúrate de poner la carpeta 'files' dentro de la carpeta 'dags'.")
    except Exception as e:
        print(f"Error en la extracción a Bronze: {e}")
        raise e



# DEFINICIÓN DEL DAG
with DAG(
        dag_id='ingesta_trafico_bronze',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['trafico', 'bronze', 'csv', 'delta'],
) as dag:
    task_bronze = PythonOperator(
        task_id='extraer_trafico_bronze',
        python_callable=extract_and_load_bronze,
    )

    trigger_silver = TriggerDagRunOperator(
        task_id="trigger_limpieza_silver",
        trigger_dag_id="limpieza_trafico_silver",
        wait_for_completion=False,
    )

    task_bronze >> trigger_silver