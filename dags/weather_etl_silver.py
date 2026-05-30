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
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true"
}

# Rutas de almacenamiento
BRONZE_PATH = "s3://bronze/meteo_raw_delta"
SILVER_PATH = "s3://silver/meteo_delta"

def clean_and_load_silver():
    print(f"2. Leyendo datos crudos desde: {BRONZE_PATH}...")

    try:
        dt = DeltaTable(BRONZE_PATH, storage_options=STORAGE_OPTIONS_DELTA)
        meteo = dt.to_pandas()
    except Exception as e:
        raise ValueError(f"Error leyendo la tabla en Bronze. Detalles: {e}")

    print("Iniciando transformaciones para la capa Silver...")

    # Transformaciones de Fechas
    meteo["time"] = pd.to_datetime(meteo["time"])
    meteo["Data"] = meteo["time"].dt.date
    meteo["Hora_dia"] = meteo["time"].dt.hour

    # Asignación de turnos
    def asignar_turno(hora):
        if 6 <= hora < 14:
            return "Matí"
        elif 14 <= hora < 22:
            return "Tarda"
        else:
            return "Nit"

    meteo["Descripcio_torn"] = meteo["Hora_dia"].apply(asignar_turno)

    # Agrupación por Día y Turno
    print("Agrupando datos meteorológicos por turno...")
    meteo_turno = meteo.groupby(["Data", "Descripcio_torn"]).agg({
        "temperature_2m": "mean",
        "precipitation": "sum",
        "rain": "sum",
        "wind_speed_10m": "mean",
        "weather_code": "max"
    }).reset_index()

    # Añadimos metadatos de transformación
    meteo_turno['etl_fecha_procesamiento'] = pd.Timestamp.now(tz='UTC')

    print(f"Transformación completada. Filas agregadas: {len(meteo_turno)}")

    # Guardar en Silver
    try:
        fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
        if not fs.exists("silver"):
            fs.mkdir("silver")

        tabla_arrow = pa.Table.from_pandas(meteo_turno)
        write_deltalake(
            table_or_uri=SILVER_PATH,
            data=tabla_arrow,
            mode="overwrite",
            schema_mode="overwrite",
            storage_options=STORAGE_OPTIONS_DELTA,
        )
        print(f"Carga agregada finalizada exitosamente en: {SILVER_PATH}")

    except Exception as e:
        print(f"Error guardando en Silver: {e}")
        raise e


# DEFINICIÓN DEL DAG SILVER
with DAG(
        dag_id='limpieza_meteo_silver',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['meteo', 'silver', 'delta'],
) as dag:
    task_silver = PythonOperator(
        task_id='transformar_meteo_silver',
        python_callable=clean_and_load_silver,
    )

    task_silver