
"""Entrenamiento, ajuste y registro de modelos HYROX con MLflow."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datos_prep import preparar_datos


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "datos" / "datos_limp" / "hyrox_muestra_10000.parquet"
EVIDENCIAS = BASE_DIR / "evidencias"
MLFLOW_DIR = EVIDENCIAS / "mlflow"
MODELOS_DIR = BASE_DIR / "modelos"

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
MODELOS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCIAS.mkdir(parents=True, exist_ok=True)

DB_PATH = MLFLOW_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
mlflow.set_experiment("Actividad5_HYROX")


# Variables utilizadas
NUMERICAS = ["year"]
CATEGORICAS = [
    "season",
    "city",
    "sex",
    "age_group",
    "division",
    "nationality",
]
PREDICTORES = NUMERICAS + CATEGORICAS
OBJETIVO = "finish_seconds"
GRUPO = "event_id"


def crear_preprocesador():
    """Imputa valores ausentes y codifica variables categóricas."""
    procesador_numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
    ])

    procesador_categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numericas", procesador_numerico, NUMERICAS),
        ("categoricas", procesador_categorico, CATEGORICAS),
    ])


def evaluar(modelo, X_test, y_test):
    """Calcula métricas estandarizadas de regresión."""
    predicciones = modelo.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, predicciones),
        "RMSE": mean_squared_error(y_test, predicciones) ** 0.5,
        "R2": r2_score(y_test, predicciones),
    }, predicciones


def registrar_modelo(nombre, modelo, metricas, parametros):
    """Registra parámetros, métricas y modelo en MLflow."""
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)
        mlflow.log_param("dataset", DATA_PATH.name)
        mlflow.log_param("variable_objetivo", OBJETIVO)
        mlflow.log_param("separacion_por_grupos", GRUPO)
        mlflow.sklearn.log_model(modelo, artifact_path="modelo")


def main():
    df = preparar_datos(DATA_PATH)

    X = df[PREDICTORES]
    y = df[OBJETIVO]
    grupos = df[GRUPO]

    # Separar por evento para reducir el riesgo de fuga de datos.
    separador = GroupShuffleSplit(
        n_splits=1, test_size=0.195, random_state=42
    )
    idx_train, idx_test = next(
        separador.split(X, y, groups=grupos)
    )

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    grupos_train = grupos.iloc[idx_train]

    cv = GroupKFold(n_splits=3)
    resultados = []
    modelos_evaluados = {}

    # Modelo 1: regresión lineal.
    modelo_lineal = Pipeline([
        ("preparacion", crear_preprocesador()),
        ("regresor", LinearRegression()),
    ])

    scores_lineal = cross_val_score(
        modelo_lineal,
        X_train,
        y_train,
        cv=cv,
        groups=grupos_train,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    modelo_lineal.fit(X_train, y_train)
    metricas_lineal, pred_lineal = evaluar(
        modelo_lineal, X_test, y_test
    )

    registrar_modelo(
        "Regresion_Lineal",
        modelo_lineal,
        {
            **metricas_lineal,
            "CV_MAE": -scores_lineal.mean(),
        },
        {"algoritmo": "LinearRegression"},
    )

    resultados.append({
        "modelo": "Regresión lineal",
        **metricas_lineal,
        "CV_MAE": -scores_lineal.mean(),
    })
    modelos_evaluados["Regresión lineal"] = modelo_lineal

    # Modelo 2: Random Forest con búsqueda de hiperparámetros.
    modelo_rf = Pipeline([
        ("preparacion", crear_preprocesador()),
        ("regresor", RandomForestRegressor(random_state=42, n_jobs=1)),
    ])

    parametros = {
        "regresor__n_estimators": [100],
        "regresor__max_depth": [None, 20],
        "regresor__min_samples_leaf": [1, 3],
    }

    busqueda = GridSearchCV(
        estimator=modelo_rf,
        param_grid=parametros,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    busqueda.fit(X_train, y_train, groups=grupos_train)
    mejor_rf = busqueda.best_estimator_

    metricas_rf, pred_rf = evaluar(mejor_rf, X_test, y_test)
    mejores_parametros = {
        clave: str(valor)
        for clave, valor in busqueda.best_params_.items()
    }

    registrar_modelo(
        "Random_Forest_GridSearch",
        mejor_rf,
        {
            **metricas_rf,
            "CV_MAE": -busqueda.best_score_,
        },
        {
            "algoritmo": "RandomForestRegressor",
            **mejores_parametros,
        },
    )

    resultados.append({
        "modelo": "Random Forest",
        **metricas_rf,
        "CV_MAE": -busqueda.best_score_,
    })
    modelos_evaluados["Random Forest"] = mejor_rf

    # Guardar resultados y predicciones.
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(
        EVIDENCIAS / "comparacion_modelos.csv", index=False
    )

    pd.DataFrame({
        "real_seconds": y_test.to_numpy(),
        "pred_lineal_seconds": pred_lineal,
        "pred_random_forest_seconds": pred_rf,
    }).to_csv(
        EVIDENCIAS / "predicciones_test.csv", index=False
    )

    # Guardar el modelo con menor MAE en el conjunto de prueba.
    mejor_nombre = tabla.loc[tabla["MAE"].idxmin(), "modelo"]
    mejor_modelo = modelos_evaluados[mejor_nombre]

    joblib.dump(
        mejor_modelo,
        MODELOS_DIR / "mejor_modelo_hyrox.joblib",
    )

    print("\nResultados:")
    print(tabla.round(4).to_string(index=False))
    print("\nMejores parámetros de Random Forest:")
    print(busqueda.best_params_)
    print(f"\nModelo seleccionado: {mejor_nombre}")
    print(f"Experimentos MLflow: {MLFLOW_DIR}")


if __name__ == "__main__":
    main()
