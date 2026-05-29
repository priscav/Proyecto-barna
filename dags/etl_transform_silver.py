import os
import pandas as pd
import pyarrow as pa
import s3fs
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from deltalake.writer import write_deltalake
from deltalake import DeltaTable

# CONFIGURACIONES Y CONSTANTES GLOBALES
MINIO_ENDPOINT = "http://minio:9000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")
TARGET_BUCKET = "silver"

# Configuracion de MinIO
STORAGE_OPTIONS_DELTA = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true"
}

def clean_and_load_delta(**context):

    # EXTRACCIÓN (LEER DESDE BRONZE - DELTA)
    bronze_file_path = context['dag_run'].conf.get('bronze_file_path')

    if not bronze_file_path:
        ds = context['ds']
        bronze_file_path = f"s3://bronze/accidentes_raw_{ds}_delta"
        print(f"No se recibió conf manual. Usando ruta por defecto: {bronze_file_path}")

    print(f"1. Leyendo datos crudos desde: {bronze_file_path}")

    try:
        dt = DeltaTable(bronze_file_path, storage_options=STORAGE_OPTIONS_DELTA)
        df = dt.to_pandas()
    except Exception as e:
        raise ValueError(f"Error al leer la tabla Delta en MinIO ({bronze_file_path}). Detalles: {e}")

    # 2. LÓGICA DE LIMPIEZA
    print("2. Iniciando limpieza de datos...")

    # Unificar la columna 'causa'
    columnas_causa = ['descripcio_causa_mediata', 'cause conductor', 'causa conductor', 'descripcio_causa']
    cols_presentes = [col for col in columnas_causa if col in df.columns]

    if cols_presentes:
        df['causa'] = df[cols_presentes].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=cols_presentes)

    # Transformar valores nulos o vacíos en 'causa' a 'Altres'
    if 'causa' in df.columns:
        df['causa'] = df['causa'].fillna('Altres').astype(str).str.strip()
        df['causa'] = df['causa'].replace(['', 'nan', 'None'], 'Altres')

    # Limpieza general
    df = df.drop_duplicates()

    if "nom_districte" in df.columns:
        df = df[df["nom_districte"] != "Desconegut"]

    if "codi_districte" in df.columns:
        df["codi_districte"] = pd.to_numeric(df["codi_districte"], errors='coerce')
        df = df[df["codi_districte"] != -1]
        df = df.dropna(subset=["codi_districte"])

    #Eliminación de nulos en columnas geográficas
    geo_columns = ["codi_districte", "nom_districte", "nom_barri", "codi_barri"]
    for col in geo_columns:
        if col in df.columns:
            df = df.dropna(subset=[col])

    if "latitud_wgs84" in df.columns and "longitud_wgs84" in df.columns:
        df = df.dropna(subset=["latitud_wgs84", "longitud_wgs84"])

    #Poner fecha completa
    if set(['nk_any', 'mes_any', 'dia_mes']).issubset(df.columns):
        df['data'] = pd.to_datetime({
            'year': pd.to_numeric(df['nk_any'], errors='coerce'),
            'month': pd.to_numeric(df['mes_any'], errors='coerce'),
            'day': pd.to_numeric(df['dia_mes'], errors='coerce')
        }, errors='coerce')

        df['data'] = df['data'].dt.date

    #Metadatos
    df['etl_fecha_ingesta'] = pd.Timestamp.now(tz='UTC')

    # CARGA (GUARDAR EN SILVER - DELTA)
    print(f"Filas después de limpiar: {len(df)}")
    print("Valores nulos restantes por columna:")
    print(df.isnull().sum())
    print(f"3. Guardando tabla Delta limpia en el bucket '{TARGET_BUCKET}'...")

    # Asegurar que el bucket Silver existe en MinIO
    fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
    if not fs.exists(TARGET_BUCKET):
        fs.mkdir(TARGET_BUCKET)

    delta_table_path = f"s3://{TARGET_BUCKET}/accidentes_delta"

    try:
        # Convertir a una tabla PyArrow
        tabla_arrow = pa.Table.from_pandas(df)

        write_deltalake(
            table_or_uri=delta_table_path,
            data=tabla_arrow,
            mode="overwrite",
            schema_mode="overwrite",
            storage_options=STORAGE_OPTIONS_DELTA,
            configuration={
                "delta.minReaderVersion": "2",
                "delta.minWriterVersion": "2"
            }
        )
        print(f"Carga en formato Delta finalizada exitosamente en: {delta_table_path}")
    except Exception as e:
        print(f"Error guardando en Delta: {e}")
        raise e

# DEFINICIÓN DEL DAG

with DAG(
        dag_id='02_limpieza_y_carga_silver',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['silver', 'delta', 'minio'],
) as dag:
    tarea_delta = PythonOperator(
        task_id='clean_and_load_to_delta',
        python_callable=clean_and_load_delta,
    )

    trigger_gold = TriggerDagRunOperator(
        task_id='trigger_capa_gold',
        trigger_dag_id='03_transformacion_y_carga_gold',
        wait_for_completion=False,
    )

    tarea_delta >> trigger_gold