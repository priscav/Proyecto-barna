import os
import pandas as pd
import numpy as np
import joblib
import tempfile
import mlflow
import mlflow.sklearn
import pyarrow as pa
import s3fs
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from deltalake import DeltaTable
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# CONFIGURACIONES GLOBALES
MINIO_ENDPOINT = "http://minio:9000"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")

STORAGE_OPTIONS_DELTA = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true"
}

os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT

RUTA_ACCIDENTES = "s3://silver/accidentes_delta"
RUTA_METEO = "s3://silver/meteo_delta"
RUTA_TRAFICO = "s3://silver/trafico_delta"


def entrenar_modelo_binario():
    print("Cargando tablas Delta desde la capa Silver de MinIO...")
    df_accidentes = DeltaTable(RUTA_ACCIDENTES, storage_options=STORAGE_OPTIONS_DELTA).to_pandas()
    df_meteo = DeltaTable(RUTA_METEO, storage_options=STORAGE_OPTIONS_DELTA).to_pandas()
    df_trafico = DeltaTable(RUTA_TRAFICO, storage_options=STORAGE_OPTIONS_DELTA).to_pandas()

    df_accidentes.columns = df_accidentes.columns.str.lower()
    df_meteo.columns = df_meteo.columns.str.lower()
    df_trafico.columns = df_trafico.columns.str.lower()

    # Correccion para asegurar que la columna temporal coincida en los cruces
    if 'fecha' in df_trafico.columns:
        df_trafico = df_trafico.rename(columns={'fecha': 'data'})

    df_accidentes["data"] = pd.to_datetime(df_accidentes["data"])
    df_meteo["data"] = pd.to_datetime(df_meteo["data"])
    df_trafico["data"] = pd.to_datetime(df_trafico["data"])

    print("Agregando y cruzando datasets de accidentes, meteo y trafico...")
    df_agg = (
        df_accidentes.groupby(["codi_districte", "nom_districte", "data", "descripcio_torn"])
        .agg(
            num_accidentes=("numero_expedient", "count"),
            dia_semana=("descripcio_dia_setmana", "first"),
            mes=("mes_any", "first"),
            dia_mes=("dia_mes", "first"),
        )
        .reset_index()
    )

    df_agg = df_agg.merge(df_meteo, on=["data", "descripcio_torn"], how="left")
    df_agg = df_agg.merge(df_trafico, on=["data", "descripcio_torn", "codi_districte"], how="left")

    media_por_turno = df_agg.groupby("descripcio_torn")["pct_congestion"].transform("mean")
    df_agg["pct_congestion"] = df_agg["pct_congestion"].fillna(media_por_turno)

    print("Creando variable objetivo binaria basada en percentil 75...")
    q75 = df_agg['num_accidentes'].quantile(0.75)
    df_agg['hay_accidente'] = (df_agg['num_accidentes'] > q75).astype(int)

    print("Codificando variables categoricas...")
    le_turno = LabelEncoder()
    le_dia = LabelEncoder()
    df_agg['turno_enc'] = le_turno.fit_transform(df_agg['descripcio_torn'])
    df_agg['dia_enc'] = le_dia.fit_transform(df_agg['dia_semana'])

    features = [
        "codi_districte", "turno_enc", "dia_enc", "mes", "dia_mes",
        "temperature_2m", "precipitation", "rain", "wind_speed_10m",
        "weather_code", "pct_congestion"
    ]

    X = df_agg[features]
    y = df_agg['hay_accidente']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Aplicando escalado de caracteristicas con StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Aplicando SMOTE para balancear el conjunto de entrenamiento...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print("Configurando MLflow y entrenando el modelo Random Forest...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("rf_binario_accidentes")

    with mlflow.start_run(run_name="RandomForest_Binario"):
        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced'
        )
        rf_model.fit(X_train_smote, y_train_smote)

        y_pred = rf_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(rf_model, artifact_path="modelo_rf", registered_model_name="RiesgoAccidentesBinario")

        print("Guardando artefactos extra (Escalador y Encoders) en MinIO...")
        fs = s3fs.S3FileSystem(key=ACCESS_KEY, secret=SECRET_KEY, client_kwargs={"endpoint_url": MINIO_ENDPOINT})

        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump(scaler, os.path.join(tmpdir, "scaler_binario.pkl"))
            joblib.dump(le_turno, os.path.join(tmpdir, "le_turno.pkl"))
            joblib.dump(le_dia, os.path.join(tmpdir, "le_dia.pkl"))

            for archivo in os.listdir(tmpdir):
                fs.put(os.path.join(tmpdir, archivo), f"s3://silver/modelos/{archivo}")

    print("Proceso de entrenamiento finalizado correctamente.")


with DAG(
        dag_id='entrenamiento_modelo',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['mlflow', 'entrenamiento'],
) as dag:
    task_entrenar = PythonOperator(
        task_id='entrenar_random_forest_binario',
        python_callable=entrenar_modelo_binario,
    )