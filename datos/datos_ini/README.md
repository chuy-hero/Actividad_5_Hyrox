# Datos originales

El conjunto de datos original utilizado en esta actividad se obtiene de Kaggle:

https://www.kaggle.com/datasets/devanshk09/complete-hyrox-dataset-with-splits-20172026

Debido a su tamaño aproximado de 20.18 GB, los archivos originales no se incluyen en el paquete de entrega.

Para descargar la versión más reciente mediante Python:

    import kagglehub

    path = kagglehub.dataset_download(
        "devanshk09/complete-hyrox-dataset-with-splits-20172026"
    )

    print("Path to dataset files:", path)

La descarga puede requerir configuración y autenticación de Kaggle.

Para reproducir el trabajo, consulta el notebook fuentes/entrena.ipynb y los datos procesados incluidos en la entrega.
