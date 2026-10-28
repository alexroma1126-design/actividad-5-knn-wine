from pathlib import Path
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELO = ROOT / "models" / "knn_wine_final.joblib"
RESULTADOS = ROOT / "outputs" / "resultados"

print("\n=== FASE 6: PREDICCION DE UN NUEVO VINO ===")

# 1. Cargar modelo entrenado
modelo = joblib.load(MODELO)

print("\n[1] Modelo recuperado")
print("Tipo:", type(modelo).__name__)
print("Vecinos:", modelo.named_steps["knn"].n_neighbors)

# 2. Crear una observacion sintetica
muestra = {
    "alcohol": 13.50,
    "malic_acid": 1.90,
    "ash": 2.50,
    "alcalinity_of_ash": 17.00,
    "magnesium": 105.00,
    "total_phenols": 2.60,
    "flavanoids": 2.90,
    "nonflavanoid_phenols": 0.29,
    "proanthocyanins": 1.80,
    "color_intensity": 5.40,
    "hue": 1.05,
    "od280/od315_of_diluted_wines": 3.10,
    "proline": 1000.00,
}

nuevo_vino = pd.DataFrame([muestra])

# 3. Verificar columnas y orden
columnas = list(modelo.feature_names_in_)

assert set(nuevo_vino.columns) == set(columnas)
nuevo_vino = nuevo_vino[columnas]

print("\n[2] Caracteristicas del vino sintetico")
print(nuevo_vino.T.to_string(header=False))

# 4. Predecir clase
prediccion = int(modelo.predict(nuevo_vino)[0])

# Fracciones de votos de los vecinos
fracciones = modelo.predict_proba(nuevo_vino)[0]
clases = modelo.named_steps["knn"].classes_

print("\n[3] RESULTADO DE LA PREDICCION")
print("Clase predicha:", prediccion)

print("\n[4] DISTRIBUCION DE VOTOS")
for clase, fraccion in zip(clases, fracciones):
    print(f"Clase {clase}: {fraccion * 100:.2f}%")

# 5. Guardar resultados
resultado = {
    "tipo": "ejemplo_sintetico",
    "clase_predicha": prediccion,
    "k": modelo.named_steps["knn"].n_neighbors,
    "fraccion_votos": {
        str(int(clase)): float(fraccion)
        for clase, fraccion in zip(clases, fracciones)
    },
    "nota": "La etiqueta real es desconocida. No es una evaluacion."
}

with open(
    RESULTADOS / "prediccion_nuevo_vino.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(resultado, archivo, indent=4)

nuevo_vino.to_csv(
    RESULTADOS / "ejemplo_nuevo_vino.csv",
    index=False
)

print("\n[5] VALIDACIONES")
print("Modelo cargado sin reentrenar: OK")
print("Columnas verificadas: OK")
print("Prediccion ejecutada: OK")
print("Resultados guardados: OK")

print("\n=== FASE 6 FINALIZADA ===")
