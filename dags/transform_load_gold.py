import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import s3fs

# CONFIGURACIONES Y CONSTANTES GLOBALES
MINIO_ENDPOINT = "http://minio:9000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")
GOLD_BUCKET = "gold"


def update_or_create_bucket_gold():
    print(f"Verificando existencia del bucket: '{GOLD_BUCKET}'...")

    fs = s3fs.S3FileSystem(
        key=ACCESS_KEY,
        secret=SECRET_KEY,
        client_kwargs={"endpoint_url": MINIO_ENDPOINT}
    )

    if not fs.exists(GOLD_BUCKET):
        print(f"El bucket '{GOLD_BUCKET}' no existe. Creando bucket gold")
        fs.mkdir(GOLD_BUCKET)
        print(f"Bucket '{GOLD_BUCKET}' creado.")
    else:
        print(f"El bucket '{GOLD_BUCKET}' ya existe. Todo listo para dbt.")


# DEFINICIÓN DEL DAG
with DAG(
        dag_id='03_transformacion_y_carga_gold',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['gold', 'dbt', 'dremio', 'minio'],
) as dag:

    # TAREAS
    task_infra = PythonOperator(
        task_id='update_or_create_bucket_gold',
        python_callable=update_or_create_bucket_gold,
    )
    task_dbt = BashOperator(
        task_id='transformar_con_dbt',
        bash_command='cd /opt/airflow/dbt_project && dbt run --profiles-dir . --full-refresh',
    )

    task_infra >> task_dbt