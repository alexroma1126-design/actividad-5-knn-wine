from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wine.csv"
RESULTADOS = ROOT / "outputs" / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos
df = pd.read_csv(DATA)

X = df.drop(columns=["target"])
y = df["target"]

print("\n=== FASE 3: PRIMER CLASIFICADOR KNN ===")

print("\n[1] Variables")
print("X:", X.shape)
print("y:", y.shape)

# 2. Division estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print("\n[2] Division entrenamiento/prueba")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

assert X_train.index.intersection(X_test.index).empty

print("\n[3] Distribucion de clases")

distribucion = pd.DataFrame({
    "Entrenamiento": y_train.value_counts().sort_index(),
    "Prueba": y_test.value_counts().sort_index(),
})

print(distribucion.to_string())

distribucion.to_csv(
    RESULTADOS / "division_estratificada.csv"
)

# 3. Pipeline: escalado + KNN
modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

print("\n[4] Configuracion del modelo")
print(modelo)

# 4. Validacion cruzada sobre entrenamiento
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_val_score(
    modelo,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
)

print("\n[5] Validacion cruzada: K=5")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print(f"\nAccuracy promedio: {scores.mean():.4f}")
print(f"Desviacion estandar: {scores.std():.4f}")

# 5. Guardar resultados
pd.DataFrame({
    "fold": np.arange(1, 6),
    "accuracy": scores,
}).to_csv(
    RESULTADOS / "baseline_knn_cv.csv",
    index=False,
)

print("\n[6] Verificaciones")
print("Sin solapamiento entrenamiento/prueba: OK")
print("Escalado dentro del pipeline: OK")
print("Datos de prueba sin evaluar: OK")

print("\n=== FASE 3 FINALIZADA ===")
