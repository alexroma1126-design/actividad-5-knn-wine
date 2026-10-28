from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
DESTINO = ROOT / "notebooks" / "Actividad_5_KNN_Wine.ipynb"
celdas = []

def agregar(tipo, contenido):
    numero = len(celdas) + 1
    celda = {
        "cell_type": tipo,
        "id": f"celda-{numero:02d}",
        "metadata": {},
        "source": textwrap.dedent(contenido).strip() + "\n",
    }
    if tipo == "code":
        celda["execution_count"] = None
        celda["outputs"] = []
    celdas.append(celda)

agregar("markdown", """
# Actividad 5 — Clasificación supervisada con KNN

**Maestría en Ciencia de Datos — Fundamentos de Inteligencia Artificial**

## Objetivo
Implementar y evaluar un clasificador K-Nearest Neighbors (KNN),
justificando la selección del hiperparámetro K.

**Dataset:** Wine Recognition (UCI Machine Learning Repository,
distribuido mediante scikit-learn).

- 178 observaciones.
- 13 características químicas.
- 3 clases.
- Clasificación supervisada multiclase.

**Metodología:** EDA, división estratificada, estandarización,
validación cruzada, selección de K y evaluación final.
""")

agregar("code", """
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

ROOT = Path.cwd()
if not (ROOT / "data" / "wine.csv").exists():
    ROOT = ROOT.parent

DATA = ROOT / "data" / "wine.csv"
assert DATA.is_file(), f"No se encontro el dataset: {DATA}"

df = pd.read_csv(DATA)
print("Dataset cargado:", df.shape)
""")

agregar("markdown", """
## 1. Análisis exploratorio

Se estudia la calidad del conjunto de datos, la distribución
de clases y la separación visual de las observaciones.
""")

agregar("code", """
print("Registros:", len(df))
print("Duplicados exactos:", df.duplicated().sum())
print("Valores faltantes:", df.isna().sum().sum())
print("\\nDistribucion por clase:")
print(df["target"].value_counts().sort_index())
print("\\nEstadisticas descriptivas:")
display(df.describe().round(3))
""")

agregar("code", """
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

df["target"].value_counts().sort_index().plot.bar(ax=axes[0])
axes[0].set_title("Distribucion de clases")
axes[0].set_xlabel("Clase")
axes[0].set_ylabel("Registros")
axes[0].tick_params(axis="x", rotation=0)

sns.scatterplot(
    data=df,
    x="flavanoids",
    y="color_intensity",
    hue="target",
    palette="deep",
    ax=axes[1]
)
axes[1].set_title("Separacion visual de clases")

plt.tight_layout()
plt.show()
""")

agregar("markdown", """
## 2. Preparación de los datos

Se utiliza una división estratificada 80/20 con semilla 42.

El escalado se integra dentro de un Pipeline para garantizar
que los parámetros de estandarización se calculen únicamente
con los datos de entrenamiento de cada ajuste.

La prueba final permanecerá reservada durante la selección de K.
""")

agregar("code", """
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

assert X_train.index.intersection(X_test.index).empty

print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)
print("\\nDistribucion de entrenamiento:")
print(y_train.value_counts().sort_index())
print("\\nDistribucion de prueba:")
print(y_test.value_counts().sort_index())
""")

agregar("markdown", """
## 3. Selección del mejor K

Se evalúan 60 valores de K mediante validación cruzada
estratificada de cinco particiones.

**Criterio:** maximizar el accuracy promedio de validación.

En caso de empate numérico, seleccionar el menor K.
Los datos de prueba no intervienen en esta selección.
""")

agregar("code", """
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

resultados = []

for k in range(1, 61):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="accuracy", n_jobs=1
    )

    resultados.append({
        "k": k,
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std())
    })

tabla_k = pd.DataFrame(resultados)
maximo = tabla_k["accuracy_mean"].max()

candidatos = tabla_k[
    np.isclose(
        tabla_k["accuracy_mean"], maximo,
        rtol=0, atol=1e-12
    )
]

mejor_k = int(candidatos["k"].min())

print("Mejor K:", mejor_k)
print(f"Accuracy CV: {maximo:.4%}")
print("\\nCinco mejores configuraciones:")
display(
    tabla_k.sort_values(
        ["accuracy_mean", "k"],
        ascending=[False, True]
    ).head(5)
)
""")

agregar("code", """
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    tabla_k["k"],
    tabla_k["accuracy_mean"],
    marker="o",
    markersize=3
)
ax.axvline(
    mejor_k,
    color="red",
    linestyle="--",
    label=f"Mejor K = {mejor_k}"
)

ax.set_title("Seleccion de K mediante validacion cruzada")
ax.set_xlabel("Numero de vecinos K")
ax.set_ylabel("Accuracy promedio")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
""")

agregar("markdown", """
## 4. Evaluación final

Una vez seleccionado K, el modelo se ajusta utilizando
todo el conjunto de entrenamiento.

Se evalúa sobre las 36 observaciones reservadas, utilizando
accuracy, balanced accuracy, precision, recall, F1-score
y una matriz de confusión.

El conjunto de prueba no se utiliza para reajustar K.
""")

agregar("code", """
modelo_final = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=mejor_k))
])

modelo_final.fit(X_train, y_train)
y_pred = modelo_final.predict(X_test)

metricas = {
    "accuracy": accuracy_score(y_test, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "precision_macro": precision_score(y_test, y_pred, average="macro"),
    "recall_macro": recall_score(y_test, y_pred, average="macro"),
    "f1_macro": f1_score(y_test, y_pred, average="macro")
}

print("METRICAS FINALES")
for nombre, valor in metricas.items():
    print(f"{nombre}: {valor:.4f}")

print("\\nREPORTE DE CLASIFICACION")
print(classification_report(y_test, y_pred, digits=4))

print("Aciertos:", int((y_test.to_numpy() == y_pred).sum()))
print("Errores:", int((y_test.to_numpy() != y_pred).sum()))
""")

agregar("code", """
fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    cmap="Blues",
    values_format="d",
    ax=ax,
    colorbar=False
)

ax.set_title(f"Matriz de confusion - KNN (K={mejor_k})")
plt.tight_layout()
plt.show()
""")

agregar("markdown", """
## 5. Predicción de una observación nueva

Se construye un ejemplo sintético con 13 características
químicas. No se conoce su etiqueta real.

La distribución mostrada corresponde a las fracciones
de votos de KNN, no necesariamente a probabilidades calibradas.
""")

agregar("code", """
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
    "proline": 1000.00
}

nuevo = pd.DataFrame([muestra])[X.columns]

clase = modelo_final.predict(nuevo)[0]
votos = modelo_final.predict_proba(nuevo)[0]

print("Clase predicha:", clase)
for etiqueta, fraccion in zip(
    modelo_final.named_steps["knn"].classes_, votos
):
    print(f"Clase {etiqueta}: {fraccion:.2%}")
""")

agregar("markdown", """
## 6. Conclusiones

En el experimento original, el mejor valor entre K=1 y K=60
fue **K=29**, con un accuracy promedio de validación cruzada
de **97.88%**.

En la prueba final se obtuvieron **35 aciertos de 36**,
equivalentes a **97.22% de accuracy**, y un F1 macro
aproximado de **97.43%**.

El único error correspondió a una observación de clase 1
clasificada como clase 0.

### Limitaciones

- El conjunto de datos tiene solamente 178 observaciones.
- El conjunto de prueba contiene 36 observaciones.
- K=29 es el mejor valor observado dentro del intervalo evaluado.
- Los resultados no garantizan el mismo rendimiento en otros datos.
- La predicción sintética no constituye una evaluación adicional.

### Mejoras futuras

Comparar otros algoritmos, ampliar la validación y realizar
experimentos con nuevos conjuntos de datos independientes.

**Fuente:** Wine Recognition Dataset, UCI Machine Learning
Repository; disponible mediante `sklearn.datasets.load_wine`.
""")

notebook = {
    "cells": celdas,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

DESTINO.parent.mkdir(parents=True, exist_ok=True)

with DESTINO.open("w", encoding="utf-8") as archivo:
    json.dump(notebook, archivo, ensure_ascii=False, indent=1)

# Verificacion estructural
with DESTINO.open(encoding="utf-8") as archivo:
    validado = json.load(archivo)

assert validado["nbformat"] == 4
assert len(validado["cells"]) == len(celdas)

for celda in validado["cells"]:
    if celda["cell_type"] == "code":
        compile(celda["source"], "<notebook>", "exec")

print("\n=== NOTEBOOK GENERADO ===")
print("Ruta:", DESTINO)
print("Celdas totales:", len(celdas))
print("Celdas Python:", sum(c["cell_type"] == "code" for c in celdas))
print("Sintaxis de Python: OK")
print("Estructura JSON: OK")
