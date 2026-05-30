import os
import pandas as pd
import numpy as np
import pyarrow as pa
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
import warnings

warnings.filterwarnings('ignore')

# CONFIGURACIONES GLOBALES
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

RUTA_ACCIDENTES = "s3://silver/accidentes_delta"
RUTA_METEO = "s3://silver/meteo_delta"
RUTA_TRAFICO = "s3://silver/trafico_delta"
OUTPUT_DELTA_PATH = "s3://silver/accidentes_simulados_2026_delta"

FILAS_POR_DIA = 50


def generate_simulated_data():
    print("Cargando tablas desde MinIO para extraer estadísticas...")
    try:
        df_acc = DeltaTable(RUTA_ACCIDENTES, storage_options=STORAGE_OPTIONS_DELTA).to_pandas()
        df_meteo = DeltaTable(RUTA_METEO, storage_options=STORAGE_OPTIONS_DELTA).to_pandas()
        df_trafico = DeltaTable(RUTA_TRAFICO, storage_options=STORAGE_OPTIONS_DELTA).to_pandas()
    except Exception as e:
        raise FileNotFoundError(f"Error al leer archivos históricos en MinIO: {e}")

    df_acc.columns = df_acc.columns.str.lower()
    df_meteo.columns = df_meteo.columns.str.lower()
    df_trafico.columns = df_trafico.columns.str.lower()

    print("Calculando estadísticas descriptivas de las variables numéricas...")
    stats = {}

    # Extraer stats meteorológicos
    cols_meteo = ['temperature_2m', 'precipitation', 'rain', 'wind_speed_10m', 'weather_code']
    for col in cols_meteo:
        if col in df_meteo.columns:
            stats[col] = {
                'mean': df_meteo[col].mean(), 'std': df_meteo[col].std(),
                'min': df_meteo[col].min(), 'max': df_meteo[col].max()
            }

    # Extraer stats de tráfico
    if 'pct_congestion' in df_trafico.columns:
        stats['pct_congestion'] = {
            'mean': df_trafico['pct_congestion'].mean(), 'std': df_trafico['pct_congestion'].std(),
            'min': df_trafico['pct_congestion'].min(), 'max': df_trafico['pct_congestion'].max()
        }

    # Extraer distritos de la tabla de accidentes
    distritos = df_acc['codi_districte'].dropna().unique()
    nombres_distrito = df_acc.groupby('codi_districte')['nom_districte'].first().to_dict()

    print(f"Análisis completado. {len(stats)} variables numéricas extraídas.")

    print("Generando matriz de datos simulados para el año 2026...")
    fechas_unicas = pd.date_range(start="2026-01-01", end="2026-12-31", freq="D")
    fechas_repetidas = np.repeat(fechas_unicas, FILAS_POR_DIA)
    total_filas = len(fechas_repetidas)

    # Mapeo para generar los nombres de días y turnos en crudo (antes del LabelEncoder)
    dias_map = {0: 'Dilluns', 1: 'Dimarts', 2: 'Dimecres', 3: 'Dijous', 4: 'Divendres', 5: 'Dissabte', 6: 'Diumenge'}
    turnos_lista = ['Matí', 'Tarda', 'Nit']

    df_2026 = pd.DataFrame({
        'data': fechas_repetidas.strftime('%Y-%m-%d'),
        'mes': fechas_repetidas.month,
        'dia_mes': fechas_repetidas.day,
        'dia_semana': [dias_map[d] for d in fechas_repetidas.weekday],  # Texto esperado por le_dia.pkl
        'descripcio_torn': np.random.choice(turnos_lista, size=total_filas),  # Texto esperado por le_turno.pkl
        'codi_districte': np.random.choice(distritos, size=total_filas)
    })

    df_2026['nom_districte'] = df_2026['codi_districte'].map(nombres_distrito).fillna('')

    # Generar variables numéricas usando las distribuciones extraídas
    for col, s in stats.items():
        valores = np.random.normal(s['mean'], s['std'], size=total_filas)
        valores = np.clip(valores, s['min'], s['max'])

        # El código del clima tiene más sentido como entero
        if col == 'weather_code':
            df_2026[col] = np.round(valores).astype(int)
        else:
            df_2026[col] = np.round(valores, 4)

    print(f"Matriz de simulación finalizada. Filas generadas: {len(df_2026):,}")

    print("Guardando los datos simulados en la capa Silver de MinIO...")
    df_2026.columns = df_2026.columns.str.lower()
    df_2026 = df_2026.astype({'data': 'str'})

    tabla_arrow = pa.Table.from_pandas(df_2026)

    write_deltalake(
        table_or_uri=OUTPUT_DELTA_PATH,
        data=tabla_arrow,
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=STORAGE_OPTIONS_DELTA
    )
    print(f"Proceso de simulación concluido exitosamente. Datos guardados en: {OUTPUT_DELTA_PATH}")


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
        task_id="trigger_predicciones_bronze",
        trigger_dag_id="prediccion_bronze_2026",
        wait_for_completion=False,
    )

    task_simulation >> trigger_inference