from minio import Minio
import os

client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password123",
    secure=False
)

modelo = "modelo_gb_binario.pkl"

with open(modelo, "rb") as f:
    client.put_object(
        "bronze",
        f"modelos/{modelo}",
        f,
        length=os.path.getsize(modelo)
    )

print(f"{modelo} subido a MinIO!")