WITH raw_accidentes AS (
    SELECT * FROM {{ source('minio_datalake', 'accidentes_delta') }}
)

SELECT
    numero_expedient AS numero_expediente,

    CAST(codi_districte AS INTEGER) AS codi_districte,
    nom_districte AS nom_districte,
    CAST(codi_barri AS INTEGER) AS codi_barri,
    nom_barri AS nom_barri,
    nom_carrer AS nom_carrer,
    num_postal AS num_postal,
    CAST(longitud AS FLOAT) AS longitud,
    CAST(latitud AS FLOAT) AS latitud,

    -- Fechas y Tiempo
    CAST(nk_any AS INTEGER) AS nk_any,
    CAST(mes_any AS INTEGER) AS mes_any,
    nom_mes AS Nom_mes,
    CAST(dia_mes AS INTEGER) AS dia_mes,
    CAST(hora_dia AS INTEGER) AS hora_dia,
    descripcio_dia_setmana AS descripcio_dia_setmana,
    descripcio_torn AS descripcio_torn,
    data AS data,

    -- Detalles del Accidente
    causa AS dausa,

    -- Metadatos de Airflow
    etl_fecha_ingesta

FROM raw_accidentes