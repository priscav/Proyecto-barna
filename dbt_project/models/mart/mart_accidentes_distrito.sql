{{ config(
    materialized='table',
    datalake='minio_datalake',
    schema='gold'
) }}
WITH accidentes AS (
    SELECT * FROM {{ ref('stg_accidentes') }}
),

agrupacion AS (
    SELECT
        nk_any,
        codi_districte,
        nom_districte,
        COUNT(numero_expediente) AS total_accidentes
    FROM accidentes
    WHERE codi_districte != -1
    GROUP BY
        nk_any,
        codi_districte,
        nom_districte
)

SELECT *
FROM agrupacion
ORDER BY nk_any DESC, total_accidentes DESC