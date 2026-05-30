import os
import joblib
import pandas as pd
import pyarrow as pa
import s3fs
import mlflow
import mlflow.sklearn
from datetime import datetime
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import warnings

warnings.filterwarnings('ignore')

# CONFIGURACIONES GLOBALES
MINIO_ENDPOINT = "http://minio:9000"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")

STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true"
}

os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT


def ejecutar_inferencia_bronze():
    print("1. Cargando datos simulados (Silver)...")
    dt_sim = DeltaTable("s3://silver/accidentes_simulados_2026_delta", storage_options=STORAGE_OPTIONS)
    df_2026 = dt_sim.to_pandas()
    df_2026.columns = df_2026.columns.str.lower()

    print("2. Cargando Artefactos de MinIO...")
    fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})

    with fs.open("s3://silver/modelos/scaler_binario.pkl", "rb") as f:
        scaler = joblib.load(f)
    with fs.open("s3://silver/modelos/le_turno.pkl", "rb") as f:
        le_turno = joblib.load(f)
    with fs.open("s3://silver/modelos/le_dia.pkl", "rb") as f:
        le_dia = joblib.load(f)

    df_2026['turno_enc'] = le_turno.transform(df_2026['descripcio_torn'])
    df_2026['dia_enc'] = le_dia.transform(df_2026['dia_semana'])

    print("3. Ejecutando Inferencia con MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    modelo = mlflow.sklearn.load_model("models:/RiesgoAccidentesBinario/latest")

    features = [
        'codi_districte', 'turno_enc', 'dia_enc', 'mes', 'dia_mes',
        'temperature_2m', 'precipitation', 'rain', 'wind_speed_10m',
        'weather_code', 'pct_congestion'
    ]

    X_scaled = scaler.transform(df_2026[features])
    df_2026['prediccion_riesgo_binario'] = modelo.predict(X_scaled)
    df_2026['probabilidad_alto_riesgo'] = modelo.predict_proba(X_scaled)[:, 1]

    print("4. Guardando tabla RAW en capa BRONZE...")
    if not fs.exists("bronze"):
        fs.mkdir("bronze")

    df_2026 = df_2026.astype({"data": "str"})
    write_deltalake(
        table_or_uri="s3://bronze/predicciones_raw_2026_delta",
        data=pa.Table.from_pandas(df_2026),
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=STORAGE_OPTIONS
    )


with DAG(
        dag_id='prediccion_bronze_2026',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['bronze', 'mlflow'],
) as dag:
    task_bronze = PythonOperator(
        task_id='inferencia_raw',
        python_callable=ejecutar_inferencia_bronze,
    )

    trigger_silver = TriggerDagRunOperator(
        task_id="trigger_limpieza_silver",
        trigger_dag_id="limpieza_silver_prediccion_2026",
        wait_for_completion=False,
    )

    task_bronze >> trigger_silver