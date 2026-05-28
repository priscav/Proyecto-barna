"""
Script para predecir accidentes en datos 2026
Lee el modelo desde MinIO usando JOBLIB (no pickle)
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import boto3
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

print("\n" + "=" * 70)
print("🔮 PREDICCIONES PARA 2026")
print("=" * 70)

# ============================================================================
# CONFIGURACIÓN MINIO
# ============================================================================
MINIO_CONFIG = {
    'endpoint_url': 'http://localhost:9000',
    'aws_access_key_id': 'admin',
    'aws_secret_access_key': 'password123',
}

BUCKET = 'bronze'
MODELO_RUTA_MINIO = 'modelos/modelo_gb_binario.pkl'

# ============================================================================
# 1. CARGAR DATOS DE 2026
# ============================================================================
print("\n1️⃣ Cargando datos de 2026...")

ruta_datos_2026 = r'C:\Users\danie\Proyecto-barna\datos_2026.csv'

try:
    df_2026 = pd.read_csv(ruta_datos_2026)
    print(f"   ✅ Datos cargados: {df_2026.shape}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print(f"   ¿Ejecutaste primero generar_datos_2026.py?")
    exit()

# ============================================================================
# 2. CARGAR MODELO DESDE MINIO (CON JOBLIB)
# ============================================================================
print("\n2️⃣ Cargando modelo desde MinIO...")

try:
    # Conectar a MinIO
    cliente_minio = boto3.client('s3', **MINIO_CONFIG)

    print(f"   Conectando a MinIO: {MINIO_CONFIG['endpoint_url']}")
    print(f"   Bucket: {BUCKET}")
    print(f"   Archivo: {MODELO_RUTA_MINIO}")

    # Descargar modelo
    response = cliente_minio.get_object(Bucket=BUCKET, Key=MODELO_RUTA_MINIO)
    modelo_bytes = response['Body'].read()

    # Deserializar CON JOBLIB
    modelo = joblib.load(BytesIO(modelo_bytes))

    print(f"   ✅ Modelo cargado desde MinIO (joblib)")
    print(f"   Tipo de modelo: {type(modelo).__name__}")

except Exception as e:
    print(f"   ❌ Error cargando desde MinIO: {e}")
    print(f"\n   Opciones:")
    print(f"   1. Verifica que MinIO está corriendo: http://localhost:9000")
    print(f"   2. Verifica ruta: s3://{BUCKET}/{MODELO_RUTA_MINIO}")
    print(f"   3. Verifica credenciales: admin/password123")
    exit()

# ============================================================================
# 3. FEATURES DEL MODELO
# ============================================================================
print("\n3️⃣ Preparando features...")

FEATURES = [
    'Codi_districte', 'turno_enc', 'dia_enc', 'mes', 'dia_mes',
    'temperature_2m', 'precipitation', 'rain', 'wind_speed_10m',
    'weather_code', 'pct_congestion'
]

# Verificar que están todos
features_faltantes = [f for f in FEATURES if f not in df_2026.columns]
if features_faltantes:
    print(f"   ⚠️  Features faltantes: {features_faltantes}")
    FEATURES = [f for f in FEATURES if f in df_2026.columns]
    print(f"   Usando: {FEATURES}")

print(f"   ✅ Features: {len(FEATURES)} columnas")

# ============================================================================
# 4. HACER PREDICCIONES
# ============================================================================
print("\n4️⃣ Haciendo predicciones (esto toma un momento)...")

X = df_2026[FEATURES]

try:
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)

    df_2026['prediccion_accidente'] = predicciones
    df_2026['prob_no_accidente'] = probabilidades[:, 0]
    df_2026['prob_accidente'] = probabilidades[:, 1]

    print(f"   ✅ Predicciones generadas para {len(df_2026):,} muestras")

except Exception as e:
    print(f"   ❌ Error: {e}")
    exit()

# ============================================================================
# 5. ANÁLISIS GENERAL
# ============================================================================
print("\n5️⃣ Analizando resultados...")

accidentes = (df_2026['prediccion_accidente'] == 1).sum()
sin_accidentes = (df_2026['prediccion_accidente'] == 0).sum()

print(f"\n   📊 PREDICCIONES TOTALES:")
print(f"      Accidentes predichos: {accidentes:,} ({accidentes / len(df_2026) * 100:.2f}%)")
print(f"      Sin accidentes: {sin_accidentes:,} ({sin_accidentes / len(df_2026) * 100:.2f}%)")

print(f"\n   📈 CONFIANZA:")
print(f"      Prob promedio de accidente: {df_2026['prob_accidente'].mean():.4f}")
print(f"      Máxima probabilidad: {df_2026['prob_accidente'].max():.4f}")
print(f"      Mínima probabilidad: {df_2026['prob_accidente'].min():.4f}")
print(f"      Std Dev: {df_2026['prob_accidente'].std():.4f}")

# ============================================================================
# 6. ANÁLISIS POR MES
# ============================================================================
print(f"\n   📅 POR MES:")

meses_nombre = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

for mes in sorted(df_2026['mes'].unique()):
    datos_mes = df_2026[df_2026['mes'] == mes]
    acc = (datos_mes['prediccion_accidente'] == 1).sum()
    total = len(datos_mes)
    pct = (acc / total * 100) if total > 0 else 0
    print(f"      {meses_nombre.get(mes, f'Mes {mes}'):12} : {acc:4} accidentes / {total:5} muestras ({pct:5.2f}%)")

# ============================================================================
# 7. ANÁLISIS POR TURNO
# ============================================================================
print(f"\n   ⏰ POR TURNO:")

turnos_nombre = {0: 'Mañana', 1: 'Tarde', 2: 'Noche'}

for turno in sorted(df_2026['turno_enc'].unique()):
    datos_turno = df_2026[df_2026['turno_enc'] == turno]
    acc = (datos_turno['prediccion_accidente'] == 1).sum()
    total = len(datos_turno)
    pct = (acc / total * 100) if total > 0 else 0
    print(
        f"      {turnos_nombre.get(turno, f'Turno {turno}'):12} : {acc:4} accidentes / {total:5} muestras ({pct:5.2f}%)")

# ============================================================================
# 8. ANÁLISIS POR DISTRITO
# ============================================================================
print(f"\n   📍 POR DISTRITO (TOP 5):")

por_distrito = df_2026.groupby('Codi_districte').agg({
    'prediccion_accidente': ['sum', 'count', 'mean']
}).round(4)

por_distrito['tasa'] = por_distrito[('prediccion_accidente', 'mean')] * 100
por_distrito_sorted = por_distrito.sort_values('tasa', ascending=False)

for dist in por_distrito_sorted.head().index:
    datos_dist = df_2026[df_2026['Codi_districte'] == dist]
    acc = (datos_dist['prediccion_accidente'] == 1).sum()
    total = len(datos_dist)
    pct = (acc / total * 100) if total > 0 else 0
    print(f"      Distrito {int(dist):2} : {acc:4} accidentes / {total:5} muestras ({pct:5.2f}%)")

# ============================================================================
# 9. CASOS DE ALTO RIESGO
# ============================================================================
print(f"\n   ⚠️  ALTO RIESGO (prob > 0.7):")

alto_riesgo = df_2026[df_2026['prob_accidente'] > 0.7]
print(f"      Total: {len(alto_riesgo):,} casos ({len(alto_riesgo) / len(df_2026) * 100:.2f}%)")

if len(alto_riesgo) > 0:
    print(f"\n      TOP 10 CASOS MÁS PELIGROSOS:")
    top_casos = alto_riesgo.nlargest(10, 'prob_accidente')[
        ['mes', 'Codi_districte', 'turno_enc', 'temperature_2m', 'pct_congestion', 'prob_accidente']
    ]
    for idx, row in top_casos.iterrows():
        print(f"      Mes {int(row['mes']):2}, Dist {int(row['Codi_districte']):2}, Turno {int(row['turno_enc'])}, " +
              f"Temp {row['temperature_2m']:6.1f}°, Congestión {row['pct_congestion']:6.1f}%, " +
              f"Riesgo {row['prob_accidente']:.4f}")

# ============================================================================
# 10. GUARDAR RESULTADOS
# ============================================================================
print(f"\n6️⃣ Guardando resultados...")

ruta_csv = r'C:\Users\danie\Proyecto-barna\predicciones_2026.csv'
ruta_excel = r'C:\Users\danie\Proyecto-barna\predicciones_2026.xlsx'

df_2026.to_csv(ruta_csv, index=False)
print(f"   ✅ CSV: {ruta_csv}")

try:
    df_2026.to_excel(ruta_excel, index=False)
    print(f"   ✅ Excel: {ruta_excel}")
except:
    pass

# Guardar resumen estadístico
resumen = {
    'Métrica': [
        'Total Predicciones',
        'Accidentes Predichos',
        'Tasa Accidentes',
        'Prob Promedio',
        'Prob Máxima',
        'Prob Mínima',
        'Alto Riesgo (>0.7)'
    ],
    'Valor': [
        f"{len(df_2026):,}",
        f"{accidentes:,}",
        f"{accidentes / len(df_2026) * 100:.2f}%",
        f"{df_2026['prob_accidente'].mean():.4f}",
        f"{df_2026['prob_accidente'].max():.4f}",
        f"{df_2026['prob_accidente'].min():.4f}",
        f"{len(alto_riesgo):,} ({len(alto_riesgo) / len(df_2026) * 100:.2f}%)"
    ]
}

df_resumen = pd.DataFrame(resumen)
ruta_resumen = r'C:\Users\danie\Proyecto-barna\resumen_predicciones_2026.csv'
df_resumen.to_csv(ruta_resumen, index=False)
print(f"   ✅ Resumen: {ruta_resumen}")

# ============================================================================
# 11. CREAR VISUALIZACIONES
# ============================================================================
print(f"\n7️⃣ Creando gráficos...")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Análisis de Predicciones - Año 2026', fontsize=16, fontweight='bold')

    # 1. Distribución de probabilidades
    axes[0, 0].hist(df_2026['prob_accidente'], bins=50, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(df_2026['prob_accidente'].mean(), color='red', linestyle='--',
                       label=f'Media: {df_2026["prob_accidente"].mean():.3f}')
    axes[0, 0].set_title('Distribución de Probabilidades')
    axes[0, 0].set_xlabel('Probabilidad')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Pie de predicciones
    labels = ['Sin Accidente', 'Accidente']
    sizes = [sin_accidentes, accidentes]
    colors = ['lightgreen', 'salmon']
    axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 1].set_title('Distribución General')

    # 3. Por mes
    meses_num = sorted(df_2026['mes'].unique())
    acc_por_mes = [
        (df_2026[df_2026['mes'] == m]['prediccion_accidente'] == 1).sum()
        for m in meses_num
    ]
    axes[0, 2].bar(meses_num, acc_por_mes, color='coral', edgecolor='black')
    axes[0, 2].set_title('Accidentes por Mes')
    axes[0, 2].set_xlabel('Mes')
    axes[0, 2].set_ylabel('Cantidad')
    axes[0, 2].grid(alpha=0.3, axis='y')

    # 4. Por turno
    turnos_num = sorted(df_2026['turno_enc'].unique())
    acc_por_turno = [
        (df_2026[df_2026['turno_enc'] == t]['prediccion_accidente'] == 1).sum()
        for t in turnos_num
    ]
    axes[1, 0].bar(turnos_num, acc_por_turno, color='lightblue', edgecolor='black')
    axes[1, 0].set_title('Accidentes por Turno')
    axes[1, 0].set_xlabel('Turno (0=Mañana, 1=Tarde, 2=Noche)')
    axes[1, 0].set_ylabel('Cantidad')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # 5. Por distrito
    distritos = sorted(df_2026['Codi_districte'].unique())[:10]
    acc_por_dist = [
        (df_2026[df_2026['Codi_districte'] == d]['prediccion_accidente'] == 1).sum()
        for d in distritos
    ]
    axes[1, 1].bar(distritos, acc_por_dist, color='yellow', edgecolor='black')
    axes[1, 1].set_title('Accidentes por Distrito (Top 10)')
    axes[1, 1].set_xlabel('Distrito')
    axes[1, 1].set_ylabel('Cantidad')
    axes[1, 1].grid(alpha=0.3, axis='y')

    # 6. Probabilidad vs Congestión
    scatter = axes[1, 2].scatter(df_2026['pct_congestion'], df_2026['prob_accidente'],
                                 c=df_2026['prediccion_accidente'], cmap='RdYlGn_r',
                                 alpha=0.3, s=10)
    axes[1, 2].set_title('Probabilidad vs Congestión')
    axes[1, 2].set_xlabel('% Congestión')
    axes[1, 2].set_ylabel('Probabilidad Accidente')
    axes[1, 2].grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2])

    plt.tight_layout()
    ruta_grafico = r'C:\Users\danie\Proyecto-barna\graficos_predicciones_2026.png'
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráficos: {ruta_grafico}")
    plt.show()

except Exception as e:
    print(f"   ⚠️  Error creando gráficos: {e}")

# ============================================================================
# 12. RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("✅ PREDICCIONES COMPLETADAS")
print("=" * 70)

print(f"\n📊 ARCHIVOS GENERADOS:")
print(f"   CSV:      {ruta_csv}")
print(f"   Excel:    {ruta_excel}")
print(f"   Resumen:  {ruta_resumen}")
print(f"   Gráficos: {ruta_grafico}")

print(f"\n📈 ESTADÍSTICAS FINALES:")
print(f"   Total de muestras: {len(df_2026):,}")
print(f"   Accidentes predichos: {accidentes:,} ({accidentes / len(df_2026) * 100:.2f}%)")
print(f"   Probabilidad promedio: {df_2026['prob_accidente'].mean():.4f}")
print(f"   Casos de alto riesgo: {len(alto_riesgo):,}")

print(f"\n💡 PRÓXIMOS PASOS:")
print(f"   1. Abre: {ruta_excel}")
print(f"   2. Carga en Dremio o PgAdmin")
print(f"   3. Analiza tendencias mensuales/por turno")
print(f"   4. Implementa alertas para alto riesgo")

print("=" * 70 + "\n")

# Mostrar muestra de datos
print("📋 MUESTRA DE PREDICCIONES:")
print(df_2026[['Codi_districte', 'turno_enc', 'prob_accidente', 'prediccion_accidente']].head(10).to_string())
