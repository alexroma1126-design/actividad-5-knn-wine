from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wine.csv"
GRAFICOS = ROOT / "outputs" / "graficos"
RESULTADOS = ROOT / "outputs" / "resultados"

GRAFICOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

print("\n=== FASE 2: ANALISIS EXPLORATORIO ===")

# 1. Integridad
print("\n[1] Calidad de los datos")
print("Registros:", len(df))
print("Duplicados:", df.duplicated().sum())
print("Valores faltantes:", df.isna().sum().sum())

# 2. Distribucion de clases
print("\n[2] Distribucion de clases")

conteos = df["target"].value_counts().sort_index()
distribucion = pd.DataFrame({
    "registros": conteos,
    "porcentaje": (conteos / len(df) * 100).round(2)
})
distribucion.index.name = "clase"

print(distribucion.to_string())
distribucion.to_csv(
    RESULTADOS / "distribucion_clases.csv"
)

# 3. Estadisticas descriptivas
print("\n[3] Estadisticas por variable")

X = df.drop(columns=["target"])

resumen = X.agg(["min", "max", "mean", "std"]).T
resumen["rango"] = resumen["max"] - resumen["min"]
resumen = resumen.sort_values(
    "rango", ascending=False
)

print(resumen.round(3).to_string())

resumen.to_csv(
    RESULTADOS / "resumen_variables.csv"
)

# 4. Grafico de clases
fig, ax = plt.subplots(figsize=(8, 5))

conteos.plot.bar(ax=ax)
ax.set_title("Distribucion de clases - Wine")
ax.set_xlabel("Clase")
ax.set_ylabel("Numero de registros")
ax.tick_params(axis="x", rotation=0)

fig.tight_layout()
fig.savefig(
    GRAFICOS / "distribucion_clases.png",
    dpi=200
)
plt.close(fig)

# 5. Grafico de separacion de clases
df["clase"] = df["target"].map({
    0: "Clase 0",
    1: "Clase 1",
    2: "Clase 2"
})

fig, ax = plt.subplots(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="flavanoids",
    y="color_intensity",
    hue="clase",
    s=70,
    ax=ax
)

ax.set_title("Separacion visual de clases")
ax.set_xlabel("Flavanoides")
ax.set_ylabel("Intensidad de color")

fig.tight_layout()
fig.savefig(
    GRAFICOS / "dispersion_caracteristicas.png",
    dpi=200
)
plt.close(fig)

print("\n[4] Graficos generados")
for archivo in sorted(GRAFICOS.glob("*.png")):
    print("-", archivo.name)

print("\n=== EDA COMPLETADO ===")
