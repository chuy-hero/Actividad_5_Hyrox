
"""Funciones de carga y preparación del dataset HYROX."""

from pathlib import Path

import pandas as pd


COLUMNAS_REQUERIDAS = [
    "season",
    "year",
    "city",
    "sex",
    "age_group",
    "division",
    "nationality",
    "event_id",
    "finish_seconds",
]


def cargar_datos(ruta):
    """Carga el Parquet y comprueba que contenga las columnas necesarias."""
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {ruta}"
        )

    df = pd.read_parquet(ruta)

    faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {faltantes}"
        )

    return df


def limpiar_datos(df):
    """Limpia el objetivo y elimina registros no utilizables."""
    df = df.copy()

    # Convertir el objetivo a número.
    df["finish_seconds"] = pd.to_numeric(
        df["finish_seconds"], errors="coerce"
    )

    # Excluir tiempos ausentes o no válidos.
    df = df[
        df["finish_seconds"].notna()
        & (df["finish_seconds"] > 0)
    ].copy()

    # No imputar event_id: se necesita para separar eventos
    # y evitar mezclar un mismo evento entre entrenamiento y prueba.
    df = df[df["event_id"].notna()].copy()

    return df.reset_index(drop=True)


def preparar_datos(ruta):
    """Carga y limpia el dataset."""
    df = cargar_datos(ruta)
    return limpiar_datos(df)
