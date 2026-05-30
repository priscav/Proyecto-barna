import os
import pandas as pd
import joblib
import tempfile
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from deltalake import DeltaTable
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# CONFIGURACIÓN (Rutas internas de Docker)
MINIO_ENDPOINT = "http://minio:9000"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "airflow")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "airflow")

os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT

STORAGE_OPTIONS_DELTA = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true"
}

STORAGE_OPTIONS_PANDAS = {
    "key": ACCESS_KEY,
    "secret": SECRET_KEY,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}
}

RUTA_ACCIDENTES = "s3://silver/accidentes_delta"
RUTA_METEO = "s3://silver/meteo_delta"
RUTA_TRAFICO = "s3://silver/trafico_agregado_distrito.csv"

def entrenar_modelo():
    print("Cargando datos desde MinIO...")
    df_accidentes = DeltaTable(RUTA_ACCIDENTES, storage_options=STORAGE_OPTIONS_DELTA).to_pandas()
    df_meteo = DeltaTable(RUTA_METEO, storage_options=STORAGE_OPTIONS_DELTA).to_pandas()
    df_trafico = pd.read_csv(RUTA_TRAFICO, storage_options=STORAGE_OPTIONS_PANDAS)

    # Normalizar columnas a minúsculas
    df_accidentes.columns = df_accidentes.columns.str.lower()
    df_meteo.columns = df_meteo.columns.str.lower()
    df_trafico.columns = df_trafico.columns.str.lower()

    df_trafico = df_trafico.rename(columns={"fecha": "data"})
    df_accidentes["data"] = pd.to_datetime(df_accidentes["data"])
    df_meteo["data"] = pd.to_datetime(df_meteo["data"])
    df_trafico["data"] = pd.to_datetime(df_trafico["data"])

    print("Cruzando datasets...")
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

    p33 = df_agg["num_accidentes"].quantile(0.33)
    p66 = df_agg["num_accidentes"].quantile(0.66)

    def clasificar_riesgo(n):
        if n <= p33: return "bajo"
        elif n <= p66: return "medio"
        else: return "alto"

    df_agg["riesgo"] = df_agg["num_accidentes"].apply(clasificar_riesgo)

    le_turno, le_dia, le_target = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df_agg["turno_enc"] = le_turno.fit_transform(df_agg["descripcio_torn"])
    df_agg["dia_enc"] = le_dia.fit_transform(df_agg["dia_semana"])
    df_agg["riesgo_enc"] = le_target.fit_transform(df_agg["riesgo"])

    FEATURES = [
        "codi_districte", "turno_enc", "dia_enc", "mes", "dia_mes",
        "temperature_2m", "precipitation", "rain", "wind_speed_10m",
        "weather_code", "pct_congestion"
    ]

    df_sorted = df_agg.sort_values("data")
    split = int(len(df_sorted) * 0.8)
    X_train, X_test = df_sorted[FEATURES].iloc[:split], df_sorted[FEATURES].iloc[split:]
    y_train, y_test = df_sorted["riesgo_enc"].iloc[:split], df_sorted["riesgo_enc"].iloc[split:]

    print(" Configurando MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("rf_accidentes_barcelona")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15],
        "min_samples_leaf": [5],
        "max_features": ["sqrt"]
    }

    mejor_f1, mejor_params, mejor_modelo = 0, None, None

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(nested=True):
            modelo = RandomForestClassifier(**params, class_weight="balanced", random_state=42, n_jobs=-1)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            f1_macro = report["macro avg"]["f1-score"]
            acc = report["accuracy"]

            mlflow.log_params(params)
            mlflow.log_metrics({"f1_macro": f1_macro, "accuracy": acc})

            if f1_macro > mejor_f1:
                mejor_f1, mejor_params, mejor_modelo = f1_macro, params, modelo

    print(f"🏆 Mejor f1_macro: {mejor_f1:.3f}. Guardando modelo...")

    with mlflow.start_run(run_name="Mejor_RandomForest"):
        mlflow.log_params(mejor_params)
        mlflow.log_metric("final_f1_macro", mejor_f1)
        mlflow.sklearn.log_model(mejor_modelo, artifact_path="modelo_rf", registered_model_name="RiesgoAccidentesRF")

        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump(le_target, os.path.join(tmpdir, "le_target.pkl"))
            joblib.dump(le_turno, os.path.join(tmpdir, "le_turno.pkl"))
            joblib.dump(le_dia, os.path.join(tmpdir, "le_dia.pkl"))
            mlflow.log_artifacts(tmpdir, artifact_path="artefactos_extra")

    print(" Entrenamiento completado.")

# DEFINICIÓN DEL DAG
with DAG(
        dag_id='entrenamiento_modelo',
        schedule='@monthly' ,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['mlflow', 'entrenamiento'],
) as dag:

    task_tranning = PythonOperator(
        task_id='entrenar_random_forest',
        python_callable=entrenar_modelo,
    )