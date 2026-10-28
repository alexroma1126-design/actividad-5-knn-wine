from pathlib import Path
from sklearn.datasets import load_wine

# Directorio raiz del proyecto
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "wine.csv"

# Cargar conjunto de datos Wine
wine = load_wine(as_frame=True)
df = wine.frame.copy()

# Guardar copia reproducible
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_PATH, index=False)

print("\n=== ACTIVIDAD 5: EXPLORACION DE WINE ===")

print("\n[1] Dimensiones")
print(df.shape)

print("\n[2] Primeros cinco registros")
print(df.head().to_string())

print("\n[3] Caracteristicas predictoras")
for columna in wine.feature_names:
    print("-", columna)

print("\n[4] Distribucion de clases")
print(df["target"].value_counts().sort_index())

print("\n[5] Nombres de las clases")
print(wine.target_names)

print("\n[6] Valores faltantes")
print(df.isnull().sum().sum())

print("\n[7] Archivo generado")
print(DATA_PATH)

print("\n=== EXPLORACION FINALIZADA ===")
