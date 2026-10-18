# Actividad 5: Predicción de tiempos HYROX

## Objetivo
Evaluar modelos de regresión para estimar el tiempo neto de finalización
de una competición HYROX utilizando variables de contexto.

## Modelos
- Regresión lineal
- Random Forest Regressor

## Métricas
- MAE
- RMSE
- R²

## Validación
Separación de entrenamiento y prueba por evento y validación cruzada
GroupKFold de cinco particiones.

## Estructura
- datos/datos_ini: datos originales, sujetos a licencia.
- datos/datos_limp: datos limpios, muestra y diccionario.
- fuentes: scripts y notebook.
- evidencias/mlflow: experimentos y modelos.
- evidencias/graficas: visualizaciones.
- reportes: informe técnico.

## Reproducibilidad
Semilla aleatoria: 42.

Los resultados deben interpretarse considerando que el modelo usa
variables de contexto y no tiempos parciales ni posiciones finales.
