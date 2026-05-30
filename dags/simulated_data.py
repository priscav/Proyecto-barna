import os
import pandas as pd
import numpy as np
import pyarrow as pa
import s3fs
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
import warnings

warnings.filterwarnings('ignore')

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

#DATOS DEL BUCKET SILVER
INPUT_PATH = "s3://silver/accidentes_delta"
OUTPUT_PATH = "s3://silver/accidentes_simulados_2026_delta"

N_NEW_ROWS = 50000

dias_semana = {
    0: "Dilluns", 1: "Dimarts", 2: "Dimecres",
    3: "Dijous", 4: "Divendres", 5: "Dissabte", 6: "Diumenge"
}

meses_cat = {
    1: "Gener", 2: "Febrer", 3: "Març", 4: "Abril",
    5: "Maig", 6: "Juny", 7: "Juliol", 8: "Agost",
    9: "Setembre", 10: "Octubre", 11: "Novembre", 12: "Desembre"
}


def generate_simulated_data():
    print(f"Cargando dataset oficial desde la capa Silver: {INPUT_PATH}")

    # Leer datos reales limpios desde Silver (Delta)
    dt = DeltaTable(INPUT_PATH, storage_options=STORAGE_OPTIONS_DELTA)
    df_real = dt.to_pandas()

    print(f"Filas originales encontradas: {len(df_real)}")
    print(f"Generando {N_NEW_ROWS} filas simuladas para 2026...")

    #Muestreo masivo: conserva todas las columnas de la capa silver
    df_sim = df_real.sample(n=N_NEW_ROWS, replace=True).reset_index(drop=True)

    #Simular el año 2026
    df_sim["nk_any"] = 2026
    df_sim["mes_any"] = np.random.randint(1, 13, N_NEW_ROWS)
    df_sim["dia_mes"] = np.random.randint(1, 29, N_NEW_ROWS)
    df_sim["hora_dia"] = np.random.randint(0, 24, N_NEW_ROWS)

    df_sim["data"] = pd.to_datetime(dict(
        year=df_sim["nk_any"],
        month=df_sim["mes_any"],
        day=df_sim["dia_mes"],
        hour=df_sim["hora_dia"]
    ))

    #Aplicar diccionarios para los días y meses
    if "descripcio_dia_setmana" in df_sim.columns:
        df_sim["descripcio_dia_setmana"] = df_sim["data"].dt.weekday.map(dias_semana)
    if "nom_mes" in df_sim.columns:
        df_sim["nom_mes"] = df_sim["mes_any"].map(meses_cat)

    #Añadir variaciones (ruido) a las coordenadas geográficas
    lat_col = 'latitud_wgs84' if 'latitud_wgs84' in df_sim.columns else 'latitud'
    lon_col = 'longitud_wgs84' if 'longitud_wgs84' in df_sim.columns else 'longitud'

    if lat_col in df_sim.columns:
        df_sim[lat_col] = pd.to_numeric(df_sim[lat_col], errors='coerce') + np.random.normal(0, 0.002, N_NEW_ROWS)
    if lon_col in df_sim.columns:
        df_sim[lon_col] = pd.to_numeric(df_sim[lon_col], errors='coerce') + np.random.normal(0, 0.002, N_NEW_ROWS)

    condiciones_turno = [
        df_sim["hora_dia"].between(6, 13),
        df_sim["hora_dia"].between(14, 21)
    ]
    df_sim["descripcio_torn"] = np.select(condiciones_turno, ["Matí", "Tarda"], default="Nit")

    #Expediente simulado para mantener unicidad
    col_expediente = 'numero_expedient' if 'numero_expedient' in df_sim.columns else 'numero_expediente'
    if col_expediente in df_sim.columns:
        df_sim[col_expediente] = ["2026SIM" + str(i).zfill(6) for i in range(N_NEW_ROWS)]

    # Metadatos de la carga
    df_sim["etl_fecha_ingesta"] = pd.Timestamp.now(tz='UTC')

    print(f"   Generación completada. Total filas simuladas: {len(df_sim)}")

    # GUARDAR EN MINIO (DELTA LAKE)
    print(f"Guardando dataset simulado en MinIO...")

    df_sim["data"] = df_sim["data"].dt.date.astype(str)

    # Transformar a PyArrow
    df_sim = df_sim.astype(str)
    tabla_arrow = pa.Table.from_pandas(df_sim)

    write_deltalake(
        table_or_uri=OUTPUT_PATH,
        data=tabla_arrow,
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=STORAGE_OPTIONS_DELTA
    )

    print(f"Dataset simulado guardado exitosamente en: {OUTPUT_PATH}")

# DEFINICIÓN DEL DAG
with DAG(
        dag_id='generar_datos_simulados_2026',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['simulacion', 'silver', 'delta'],
) as dag:
    task_simulation = PythonOperator(
        task_id='generar_y_guardar_simulacion',
        python_callable=generate_simulated_data,
    )

    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_predicciones_gold",
        trigger_dag_id="prediccion_accidentes_2026",
        wait_for_completion=False,
    )

    task_simulation >> trigger_inference