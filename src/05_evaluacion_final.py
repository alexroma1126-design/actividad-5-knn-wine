from pathlib import Path
import json

import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "wine.csv"
RESULTADOS = ROOT / "outputs" / "resultados"
GRAFICOS = ROOT / "outputs" / "graficos"
MODELOS = ROOT / "models"

for carpeta in (RESULTADOS, GRAFICOS, MODELOS):
    carpeta.mkdir(parents=True, exist_ok=True)

print("\n=== FASE 5: EVALUACION FINAL KNN ===")

# 1. Recuperar configuracion validada
with open(
    RESULTADOS / "mejor_k_extendido.json",
    encoding="utf-8"
) as archivo:
    config = json.load(archivo)

mejor_k = int(config["mejor_k"])

assert config["test_evaluado"] is False
assert 1 <= mejor_k <= 60

print("\n[1] Hiperparametro seleccionado")
print("Mejor K:", mejor_k)
print(f"Accuracy CV: {config['accuracy_cv']:.4f}")

# 2. Cargar datos
df = pd.read_csv(DATA)

X = df.drop(columns=["target"])
y = df["target"]

# 3. Reproducir division original
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

assert X_train.index.intersection(X_test.index).empty
assert len(X_train) == 142
assert len(X_test) == 36

print("\n[2] Datos")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

# 4. Entrenar modelo final
modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=mejor_k)),
])

modelo.fit(X_train, y_train)

print("\n[3] Modelo final entrenado")

# 5. Evaluar datos de prueba
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)
f1_macro = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

aciertos = int((y_test.to_numpy() == y_pred).sum())
errores = len(y_test) - aciertos

print("\n[4] METRICAS FINALES")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced:.4f}")
print(f"Precision macro:   {precision:.4f}")
print(f"Recall macro:      {recall:.4f}")
print(f"F1-score macro:    {f1_macro:.4f}")
print(f"Aciertos:          {aciertos}/36")
print(f"Errores:           {errores}/36")

# 6. Reporte por clase
etiquetas = [0, 1, 2]
nombres = ["Clase 0", "Clase 1", "Clase 2"]

reporte = classification_report(
    y_test,
    y_pred,
    labels=etiquetas,
    target_names=nombres,
    digits=4,
    zero_division=0,
)

print("\n[5] REPORTE DE CLASIFICACION")
print(reporte)

(RESULTADOS / "reporte_clasificacion.txt").write_text(
    reporte,
    encoding="utf-8",
)

# 7. Matriz de confusion
cm = confusion_matrix(
    y_test, y_pred, labels=etiquetas
)

print("\n[6] MATRIZ DE CONFUSION")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=nombres,
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title(f"Matriz de confusion - KNN (K={mejor_k})")
fig.tight_layout()

fig.savefig(
    GRAFICOS / "matriz_confusion_final.png",
    dpi=250,
)
plt.close(fig)

# 8. Guardar predicciones
predicciones = pd.DataFrame({
    "indice_original": X_test.index,
    "clase_real": y_test.to_numpy(),
    "clase_predicha": y_pred,
})

predicciones["correcto"] = (
    predicciones["clase_real"]
    == predicciones["clase_predicha"]
)

predicciones.to_csv(
    RESULTADOS / "predicciones_test.csv",
    index=False,
)

# 9. Guardar metricas finales
metricas = {
    "dataset": "Wine",
    "mejor_k": mejor_k,
    "accuracy_cv": float(config["accuracy_cv"]),
    "accuracy_test": float(accuracy),
    "balanced_accuracy": float(balanced),
    "precision_macro": float(precision),
    "recall_macro": float(recall),
    "f1_macro": float(f1_macro),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "aciertos": aciertos,
    "errores": errores,
    "matriz_confusion": cm.tolist(),
}

with open(
    RESULTADOS / "metricas_finales.json",
    "w",
    encoding="utf-8",
) as archivo:
    json.dump(metricas, archivo, indent=4)

# 10. Exportar modelo
ruta_modelo = MODELOS / "knn_wine_final.joblib"

joblib.dump(modelo, ruta_modelo)

print("\n[7] ARCHIVOS GENERADOS")
print("- reporte_clasificacion.txt")
print("- predicciones_test.csv")
print("- metricas_finales.json")
print("- matriz_confusion_final.png")
print("- knn_wine_final.joblib")

print("\n=== FASE 5 COMPLETADA ===")
