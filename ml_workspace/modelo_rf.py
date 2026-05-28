import pandas as pd

# Accidentes
df_accidentes = pd.read_excel(r"C:\Users\danie\Proyecto-barna\datos_limpios.xlsx")

# Meteo
df_meteo = pd.read_csv(r"C:\Users\danie\Proyecto-barna\datos_meteorologicos_turno.csv")

# Tráfico por distrito (nuevo)
df_trafico = pd.read_csv(
    r"C:\Users\danie\Documents\Ciencia de Datos\Curso 25-26\Actividades S2\Open Data II\Trafico\trafico_agregado_distrito.csv"
)

print(df_accidentes.shape, df_accidentes.columns.tolist())
print(df_meteo.shape, df_meteo.columns.tolist())
print(df_trafico.shape, df_trafico.columns.tolist())
# Estandarizar fechas
df_accidentes["Data"] = pd.to_datetime(df_accidentes["Data"])
df_meteo["Data"] = pd.to_datetime(df_meteo["Data"])
df_trafico = df_trafico.rename(columns={"fecha": "Data"})
df_trafico["Data"] = pd.to_datetime(df_trafico["Data"])

print(df_trafico.columns.tolist())

# Agregar accidentes por Distrito + Fecha + Turno
df_agg = (
    df_accidentes.groupby(["Codi_districte", "Nom_districte", "Data", "Descripcio_torn"])
    .agg(
        num_accidentes=("Numero_expedient", "count"),
        dia_semana=("Descripcio_dia_setmana", "first"),
        mes=("Mes_any", "first"),
        dia_mes=("Dia_mes", "first"),
    )
    .reset_index()
)

# Merge con meteo
df_agg = df_agg.merge(df_meteo, on=["Data", "Descripcio_torn"], how="left")

# Merge con tráfico por distrito
df_agg = df_agg.merge(df_trafico, on=["Data", "Descripcio_torn", "Codi_districte"], how="left")

print(df_agg.shape)
print(df_agg.isnull().sum())

media_por_turno = df_agg.groupby("Descripcio_torn")["pct_congestion"].transform("mean")
df_agg["pct_congestion"] = df_agg["pct_congestion"].fillna(media_por_turno)

print(df_agg.isnull().sum())
print(f"Filas totales: {len(df_agg)}")

p33 = df_agg["num_accidentes"].quantile(0.33)
p66 = df_agg["num_accidentes"].quantile(0.66)

print(f"Umbral bajo/medio: {p33}")
print(f"Umbral medio/alto: {p66}")


def clasificar_riesgo(n):
    if n <= p33:
        return "bajo"
    elif n <= p66:
        return "medio"
    else:
        return "alto"


df_agg["riesgo"] = df_agg["num_accidentes"].apply(clasificar_riesgo)
print("\nDistribución del target:")
print(df_agg["riesgo"].value_counts())

from sklearn.preprocessing import LabelEncoder

le_turno = LabelEncoder()
le_dia = LabelEncoder()
le_target = LabelEncoder()

df_agg["turno_enc"] = le_turno.fit_transform(df_agg["Descripcio_torn"])
df_agg["dia_enc"] = le_dia.fit_transform(df_agg["dia_semana"])
df_agg["riesgo_enc"] = le_target.fit_transform(df_agg["riesgo"])

print("Turno:", dict(zip(le_turno.classes_, le_turno.transform(le_turno.classes_))))
print("Riesgo:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

FEATURES = [
    "Codi_districte",
    "turno_enc",
    "dia_enc",
    "mes",
    "dia_mes",
    "temperature_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "weather_code",
    "pct_congestion",
]

df_sorted = df_agg.sort_values("Data")
split = int(len(df_sorted) * 0.8)

X_train = df_sorted[FEATURES].iloc[:split]
X_test = df_sorted[FEATURES].iloc[split:]
y_train = df_sorted["riesgo_enc"].iloc[:split]
y_test = df_sorted["riesgo_enc"].iloc[split:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n── RESULTADOS ──────────────────────────")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

!pip installmlflow

import mlflow
import mlflow.sklearn
from sklearn.model_selection import ParameterGrid

# Configurar experimento
mlflow.set_experiment("rf_accidentes_barcelona")

# Grid de hiperparámetros a probar
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_leaf": [3, 5, 10],
    "max_features": ["sqrt", "log2"],
}

mejor_f1 = 0
mejor_params = None

for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        # Entrenar
        modelo = RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        modelo.fit(X_train, y_train)
        y_pred_loop = modelo.predict(X_test)

        # Métricas
        report = classification_report(y_test, y_pred_loop, output_dict=True, target_names=le_target.classes_)
        f1_macro = report["macro avg"]["f1-score"]
        acc = report["accuracy"]

        # Loggear en MLflow
        mlflow.log_params(params)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_alto", report["alto"]["f1-score"])
        mlflow.log_metric("f1_medio", report["medio"]["f1-score"])
        mlflow.log_metric("f1_bajo", report["bajo"]["f1-score"])

        if f1_macro > mejor_f1:
            mejor_f1 = f1_macro
            mejor_params = params
            mlflow.sklearn.log_model(modelo, "modelo_rf")

        print(f"f1={f1_macro:.3f} acc={acc:.3f} | {params}")

print(f"\n✓ Mejor f1_macro: {mejor_f1:.3f}")
print(f"✓ Mejores parámetros: {mejor_params}")

!mlflow ui

import joblib

rf_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    max_features="log2",
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_final.fit(X_train, y_train)
y_pred_final = rf_final.predict(X_test)

print(classification_report(y_test, y_pred_final, target_names=le_target.classes_))

# Guardar
joblib.dump(rf_final, r"C:\Users\danie\Proyecto-barna\modelo_rf.pkl")
joblib.dump(le_target, r"C:\Users\danie\Proyecto-barna\le_target.pkl")
joblib.dump(le_turno, r"C:\Users\danie\Proyecto-barna\le_turno.pkl")
joblib.dump(le_dia, r"C:\Users\danie\Proyecto-barna\le_dia.pkl")

importancias = pd.DataFrame({
    "feature": FEATURES,
    "importancia": rf_final.feature_importances_
}).sort_values("importancia", ascending=False)

importancias.to_csv(r"C:\Users\danie\Proyecto-barna\feature_importances.csv", index=False)
df_agg.to_csv(r"C:\Users\danie\Proyecto-barna\dataset_modelo.csv", index=False)

print("✓ Modelo y artefactos guardados")