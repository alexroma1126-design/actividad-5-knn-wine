from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
GRAFICOS = ROOT / "outputs" / "graficos"

RESULTADOS.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)

print("\n=== FASE 4: OPTIMIZACION DE K ===")

# 1. Cargar datos
df = pd.read_csv(DATA)

X = df.drop(columns=["target"])
y = df["target"]

# 2. Reproducir la division original
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# 3. Configurar validacion cruzada
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# 4. Evaluar los valores de K
resultados = []
resultados_folds = []

print("\n[1] Evaluando valores de K")
print(f"{'K':>4} {'Accuracy medio':>18} {'Desv. estandar':>18}")
print("-" * 45)

for k in range(1, 31):

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )

    media = float(np.mean(scores))
    desviacion = float(np.std(scores))

    resultados.append({
        "k": k,
        "accuracy_mean": media,
        "accuracy_std": desviacion,
    })

    for fold, score in enumerate(scores, start=1):
        resultados_folds.append({
            "k": k,
            "fold": fold,
            "accuracy": float(score),
        })

    print(f"{k:>4} {media:>18.4f} {desviacion:>18.4f}")

# 5. Convertir a DataFrame
df_resultados = pd.DataFrame(resultados)

# Seleccionar el maximo promedio
mejor_accuracy = df_resultados["accuracy_mean"].max()

# Resolver empates con tolerancia numerica
empates = df_resultados[
    np.isclose(
        df_resultados["accuracy_mean"],
        mejor_accuracy,
        rtol=0,
        atol=1e-12,
    )
]

mejor_fila = empates.sort_values("k").iloc[0]
mejor_k = int(mejor_fila["k"])

print("\n[2] MEJOR VALOR DE K")
print("K optimo:", mejor_k)
print(f"Accuracy CV: {mejor_accuracy:.4f}")
print(f"Porcentaje: {mejor_accuracy * 100:.2f}%")
print("Valores empatados:", empates["k"].tolist())

# 6. Mostrar los mejores resultados
print("\n[3] TOP 5 VALORES DE K")

top5 = df_resultados.sort_values(
    ["accuracy_mean", "k"],
    ascending=[False, True],
).head(5)

print(top5.to_string(index=False))

# 7. Guardar resultados
df_resultados.to_csv(
    RESULTADOS / "optimizacion_k.csv",
    index=False,
)

pd.DataFrame(resultados_folds).to_csv(
    RESULTADOS / "optimizacion_k_folds.csv",
    index=False,
)

configuracion = {
    "dataset": "Wine",
    "random_state": 42,
    "test_size": 0.20,
    "cv_folds": 5,
    "k_min": 1,
    "k_max": 30,
    "mejor_k": mejor_k,
    "accuracy_cv": float(mejor_accuracy),
    "criterio_empate": "menor K",
    "test_evaluado": False,
}

with open(
    RESULTADOS / "mejor_k.json",
    "w",
    encoding="utf-8",
) as archivo:
    json.dump(configuracion, archivo, indent=4)

# 8. Graficar curva K vs Accuracy
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    df_resultados["k"],
    df_resultados["accuracy_mean"],
    marker="o",
    linewidth=2,
    markersize=4,
    label="Accuracy promedio CV",
)

ax.axvline(
    mejor_k,
    color="red",
    linestyle="--",
    label=f"Mejor K = {mejor_k}",
)

ax.scatter(
    [mejor_k],
    [mejor_accuracy],
    color="red",
    s=100,
    zorder=5,
)

ax.set_title("Seleccion del mejor K - Wine")
ax.set_xlabel("Numero de vecinos K")
ax.set_ylabel("Accuracy promedio (5 folds)")
ax.set_xticks(range(1, 31))
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    GRAFICOS / "optimizacion_k.png",
    dpi=250,
)
plt.close(fig)

print("\n[4] ARCHIVOS GENERADOS")
print("- optimizacion_k.csv")
print("- optimizacion_k_folds.csv")
print("- mejor_k.json")
print("- optimizacion_k.png")

print("\n[5] CONTROL METODOLOGICO")
print("Escalado dentro de cada fold: OK")
print("Datos de prueba reservados: OK")
print("Semilla reproducible: 42")

print("\n=== FASE 4 FINALIZADA ===")
