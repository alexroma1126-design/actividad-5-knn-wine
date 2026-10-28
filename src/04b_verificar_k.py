from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "outputs" / "resultados"
GRAFICOS = ROOT / "outputs" / "graficos"

print("\n=== FASE 4B: VERIFICACION DEL MEJOR K ===")

df = pd.read_csv(ROOT / "data" / "wine.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, _, y_train, _ = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Recuperar resultados anteriores (K=1 a K=30)
anteriores = pd.read_csv(
    RESULTADOS / "optimizacion_k.csv"
)

assert anteriores["k"].tolist() == list(range(1, 31))

nuevos = []

print("\n[1] Evaluando K=31 hasta K=60")

for k in range(31, 61):

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    nuevos.append({
        "k": k,
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std())
    })

    print(f"K={k:2d} | Accuracy={scores.mean():.4f}")

# Combinar resultados
resultados = pd.concat(
    [anteriores, pd.DataFrame(nuevos)],
    ignore_index=True
).sort_values("k")

maximo = resultados["accuracy_mean"].max()

candidatos = resultados[
    np.isclose(
        resultados["accuracy_mean"],
        maximo,
        rtol=0,
        atol=1e-12
    )
]

mejor = candidatos.sort_values("k").iloc[0]
mejor_k = int(mejor["k"])

print("\n[2] RESULTADO FINAL DE LA BUSQUEDA")
print("Mejor K:", mejor_k)
print(f"Accuracy CV: {maximo:.6f}")
print(f"Porcentaje: {maximo * 100:.2f}%")
print("Empates:", candidatos["k"].tolist())

print("\n[3] TOP 10")
print(
    resultados.sort_values(
        ["accuracy_mean", "k"],
        ascending=[False, True]
    ).head(10).to_string(index=False)
)

# Guardar resultados
resultados.to_csv(
    RESULTADOS / "optimizacion_k_extendida.csv",
    index=False
)

config = {
    "dataset": "Wine",
    "rango_k": [1, 60],
    "cv_folds": 5,
    "random_state": 42,
    "mejor_k": mejor_k,
    "accuracy_cv": float(maximo),
    "criterio_empate": "menor K",
    "test_evaluado": False
}

with open(
    RESULTADOS / "mejor_k_extendido.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(config, f, indent=4)

# Nuevo grafico
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    resultados["k"],
    resultados["accuracy_mean"],
    marker="o",
    markersize=3,
    label="Accuracy CV"
)

ax.axvline(
    mejor_k,
    color="red",
    linestyle="--",
    label=f"Mejor K = {mejor_k}"
)

ax.scatter(
    [mejor_k], [maximo],
    color="red",
    s=90,
    zorder=5
)

ax.set_xlabel("Numero de vecinos K")
ax.set_ylabel("Accuracy promedio")
ax.set_title("Busqueda ampliada del mejor K - Wine")
ax.set_xticks(range(1, 61, 2))
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    GRAFICOS / "optimizacion_k_extendida.png",
    dpi=250
)

plt.close(fig)

print("\n[4] VERIFICACIONES")
print("Rango evaluado: K=1 a K=60")
print("Datos de prueba sin utilizar: OK")
print("Resultados guardados: OK")

print("\n=== FASE 4B COMPLETADA ===")
