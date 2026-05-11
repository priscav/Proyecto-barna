import pandas as pd
import requests
from datetime import datetime

# =========================
# 1. CONFIGURACIÓN
# =========================
API_BASE_URL = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search"

RESOURCE_IDS_BY_YEAR = {
    2022: "87a8aeda-d3eb-4ba5-bcad-b9ab0c296df5",
    2023: "5a040155-38b3-4b19-a4b0-c84a0618d363",
    2024: "8cfddcbe-3403-4a6c-8897-c13238da900e",
    2025: "796504f6-7602-41a7-82a9-d3bd47e68dee"
}

OUTPUT_FILE = "data_cleaned.csv"

# =========================
# 2. DESCARGA DE DATOS
# =========================
def fetch_data_from_api(api_url, resource_ids_by_year):

    print("Obteniendo datos desde Open Data BCN...")

    all_dataframes = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for year, resource_id in resource_ids_by_year.items():

        print(f"→ Descargando año {year}")

        params = {
            "resource_id": resource_id,
            "limit": 32000
        }

        response = requests.get(
            api_url,
            params=params,
            headers=headers
        )

        if response.status_code == 200:

            data = response.json()

            if data["success"]:

                df_year = pd.DataFrame(
                    data["result"]["records"]
                )

                all_dataframes.append(df_year)

                print(f"   ✔ {len(df_year)} filas")

            else:
                print(f"   ✖ Error API {year}")

        else:
            print(f"   ✖ Error conexión {year}")

    if not all_dataframes:
        raise Exception("No se pudieron descargar datos")

    df = pd.concat(
        all_dataframes,
        ignore_index=True
    )

    print(f"\nTotal filas descargadas: {len(df)}")

    return df


# =========================
# 3. LIMPIEZA Y NORMALIZACIÓN
# =========================
def clean_data(df):

    print("\nLimpiando datos...")

    # ====================================
    # ELIMINAR DUPLICADOS
    # ====================================
    df = df.drop_duplicates()

    # ====================================
    # NORMALIZAR COLUMNAS
    # ====================================

    # numero_expedient
    expediente_cols = [
        col for col in [
            "Número_expedient",
            "Numero_expedient"
        ]
        if col in df.columns
    ]

    if expediente_cols:
        df["numero_expedient"] = (
            df[expediente_cols]
            .bfill(axis=1)
            .iloc[:, 0]
        )

    # nk_Any
    year_cols = [
        col for col in [
            "NK_Any",
            "Nk_Any"
        ]
        if col in df.columns
    ]

    if year_cols:
        df["nk_Any"] = (
            df[year_cols]
            .bfill(axis=1)
            .iloc[:, 0]
        )

    # ====================================
    # UNIFICAR CAUSA
    # ====================================

    if (
        "Descripcio_causa_mediata" in df.columns
        and
        "Causa conductor" in df.columns
    ):

        df["causa"] = (
            df["Descripcio_causa_mediata"]
            .fillna(df["Causa conductor"])
        )

    elif "Descripcio_causa_mediata" in df.columns:

        df["causa"] = df["Descripcio_causa_mediata"]

    elif "Causa conductor" in df.columns:

        df["causa"] = df["Causa conductor"]

    # ====================================
    # ELIMINAR DESCONOCIDOS
    # ====================================

    if "Nom_districte" in df.columns:
        df = df[
            df["Nom_districte"] != "Desconegut"
        ]

    if "Codi_districte" in df.columns:

        df["Codi_districte"] = pd.to_numeric(
            df["Codi_districte"],
            errors="coerce"
        )

        df = df[
            df["Codi_districte"] != -1
        ]

    # ====================================
    # LIMPIAR NULOS GEO
    # ====================================

    geo_cols = [
        "Nom_districte",
        "Nom_barri",
        "Codi_barri"
    ]

    for col in geo_cols:

        if col in df.columns:
            df = df.dropna(
                subset=[col]
            )

    # ====================================
    # COORDENADAS
    # ====================================

    if (
        "Latitud_WGS84" in df.columns
        and
        "Longitud_WGS84" in df.columns
    ):

        df = df.dropna(
            subset=[
                "Latitud_WGS84",
                "Longitud_WGS84"
            ]
        )

    # ====================================
    # FECHA INGESTA
    # ====================================

    df["etl_fecha_ingesta"] = (
        datetime.now()
    )

    # ====================================
    # RENOMBRAR COLUMNAS
    # ====================================

    rename_columns = {

        "Codi_districte": "codi_districte",
        "Nom_districte": "nom_districte",

        "Codi_barri": "codi_barri",
        "Nom_barri": "nom_barri",

        "Codi_carrer": "codi_carrer",
        "Nom_carrer": "nom_carrer",

        "Num_postal": "num_postal",

        "Descripcio_dia_setmana": "descripcio_dia_setmana",

        "Mes_any": "mes_any",
        "Nom_mes": "nom_mes",

        "Dia_mes": "dia_mes",

        "Hora_dia": "hora_dia",

        "Descripcio_torn": "descripcio_torn",

        "Longitud_WGS84": "longitud",
        "Latitud_WGS84": "latitud",

        "Data": "data",

        "_id": "_id"
    }

    df = df.rename(
        columns=rename_columns
    )

    # ====================================
    # ELIMINAR COLUMNAS ANTIGUAS
    # ====================================

    columns_to_remove = [

        "Número_expedient",
        "Numero_expedient",

        "NK_Any",
        "Nk_Any",

        "Descripcio_causa_mediata",
        "Causa conductor"
    ]

    df = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    # ====================================
    # ORDEN FINAL
    # ====================================

    final_columns = [

        "numero_expedient",

        "codi_districte",
        "nom_districte",

        "codi_barri",
        "nom_barri",

        "codi_carrer",
        "nom_carrer",

        "num_postal",

        "descripcio_dia_setmana",

        "nk_Any",

        "mes_any",
        "nom_mes",

        "dia_mes",

        "hora_dia",

        "descripcio_torn",

        "longitud",
        "latitud",

        "causa",

        "data",

        "_id",

        "etl_fecha_ingesta"
    ]

    existing_columns = [
        col for col in final_columns
        if col in df.columns
    ]

    df = df[existing_columns]

    # ====================================
    # RESULTADO FINAL
    # ====================================

    print("\nColumnas finales:")
    print(df.columns.tolist())

    print(f"\nFilas finales: {len(df)}")

    return df


# =========================
# 4. GUARDAR CSV
# =========================
def save_data(df, output_file):

    df.to_csv(
        output_file,
        index=False
    )

    print(f"\nDataset guardado: {output_file}")


# =========================
# 5. PIPELINE
# =========================
def run_pipeline():

    df_raw = fetch_data_from_api(
        API_BASE_URL,
        RESOURCE_IDS_BY_YEAR
    )

    df_clean = clean_data(df_raw)

    save_data(
        df_clean,
        OUTPUT_FILE
    )

    print("\n✔ Pipeline completado")


# =========================
# EJECUCIÓN
# =========================
if __name__ == "__main__":
    run_pipeline()
