import json
import unittest
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wine.csv"
RESULTADOS = ROOT / "outputs" / "resultados"
NOTEBOOK = ROOT / "notebooks" / "Actividad_5_KNN_Wine_ejecutado.ipynb"
MODELO = ROOT / "models" / "knn_wine_final.joblib"


class TestProyectoKNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA)

        with open(
            RESULTADOS / "metricas_finales.json",
            encoding="utf-8"
        ) as archivo:
            cls.metricas = json.load(archivo)

        with open(
            RESULTADOS / "mejor_k_extendido.json",
            encoding="utf-8"
        ) as archivo:
            cls.config = json.load(archivo)

        cls.modelo = joblib.load(MODELO)

        X = cls.df.drop(columns=["target"])
        y = cls.df["target"]

        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
        ) = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42
        )

    def test_01_integridad_dataset(self):
        self.assertEqual(self.df.shape, (178, 14))
        self.assertEqual(int(self.df.isna().sum().sum()), 0)
        self.assertEqual(int(self.df.duplicated().sum()), 0)

        self.assertEqual(
            self.df["target"].value_counts().sort_index().to_dict(),
            {0: 59, 1: 71, 2: 48}
        )

    def test_02_division_datos(self):
        self.assertEqual(len(self.X_train), 142)
        self.assertEqual(len(self.X_test), 36)

        self.assertTrue(
            self.X_train.index.intersection(
                self.X_test.index
            ).empty
        )

        self.assertEqual(
            self.y_test.value_counts().sort_index().to_dict(),
            {0: 12, 1: 14, 2: 10}
        )

    def test_03_hiperparametro(self):
        tabla = pd.read_csv(
            RESULTADOS / "optimizacion_k_extendida.csv"
        )

        self.assertEqual(len(tabla), 60)
        self.assertEqual(
            tabla["k"].tolist(), list(range(1, 61))
        )

        maximo = tabla["accuracy_mean"].max()

        candidatos = tabla[
            np.isclose(
                tabla["accuracy_mean"],
                maximo,
                rtol=0,
                atol=1e-12
            )
        ]

        mejor_k = int(candidatos["k"].min())

        self.assertEqual(mejor_k, 29)
        self.assertEqual(mejor_k, self.config["mejor_k"])

        self.assertAlmostEqual(
            maximo,
            self.config["accuracy_cv"],
            places=10
        )

    def test_04_metricas_modelo(self):
        predicciones = self.modelo.predict(self.X_test)

        accuracy = accuracy_score(
            self.y_test, predicciones
        )

        f1 = f1_score(
            self.y_test,
            predicciones,
            average="macro"
        )

        matriz = confusion_matrix(
            self.y_test,
            predicciones,
            labels=[0, 1, 2]
        )

        self.assertAlmostEqual(
            accuracy,
            self.metricas["accuracy_test"],
            places=10
        )

        self.assertAlmostEqual(
            f1,
            self.metricas["f1_macro"],
            places=10
        )

        self.assertEqual(
            matriz.tolist(),
            self.metricas["matriz_confusion"]
        )

        self.assertEqual(
            int((self.y_test.to_numpy() == predicciones).sum()),
            35
        )

    def test_05_prediccion_nueva(self):
        nuevo = pd.read_csv(
            RESULTADOS / "ejemplo_nuevo_vino.csv"
        )

        with open(
            RESULTADOS / "prediccion_nuevo_vino.json",
            encoding="utf-8"
        ) as archivo:
            esperado = json.load(archivo)

        nuevo = nuevo[list(self.modelo.feature_names_in_)]

        prediccion = int(self.modelo.predict(nuevo)[0])
        votos = self.modelo.predict_proba(nuevo)[0]

        self.assertEqual(
            prediccion,
            esperado["clase_predicha"]
        )

        self.assertAlmostEqual(float(votos.sum()), 1.0)

        self.assertEqual(prediccion, 0)

    def test_06_notebook_ejecutado(self):
        nb = nbformat.read(NOTEBOOK, as_version=4)

        codigo = [
            celda for celda in nb.cells
            if celda.cell_type == "code"
        ]

        errores = [
            salida
            for celda in codigo
            for salida in celda.get("outputs", [])
            if salida.output_type == "error"
        ]

        self.assertEqual(len(nb.cells), 16)
        self.assertEqual(len(codigo), 9)
        self.assertEqual(len(errores), 0)

        self.assertTrue(
            all(
                celda.execution_count is not None
                for celda in codigo
            )
        )


if __name__ == "__main__":
    unittest.main()
