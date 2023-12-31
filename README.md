# Reconocimiento Facial con OpenCV y dlib

Este repositorio contiene dos scripts de Python que demuestran cómo detectar y reconocer caras en videos utilizando OpenCV, dlib y numpy.

## Requisitos

- Python 3.6 o superior
- OpenCV
- dlib
- numpy

Puede instalar las dependencias utilizando pip:

```bash
pip install opencv-python dlib numpy
```

## Uso

1. **Código de Detección Facial (main-script.py):**

   Este script detecta caras en un video en tiempo real o en un video pregrabado utilizando una cámara o un archivo de video.

   Para ejecutarlo:

   ```bash
   python main-script.py
   ```

2. **Código de Reconocimiento Facial (reconocimiento.py):**

   Este script detecta y reconoce caras en un video basado en una lista predefinida de embeddings y nombres de caras.

   Primero, asegúrese de tener una lista predefinida de embeddings y nombres según se describe en la sección [Cómo crear la lista de embeddings y nombres](#cómo-crear-la-lista-de-embeddings-y-nombres).

   Luego, ejecute el script:

   ```bash
   python reconocimiento.py
   ```

## Cómo crear la lista de embeddings y nombres

Para construir la lista de `face_embeddings` y `names`, sigue estos pasos:

1. **Colección de Datos:**
    - Recolecta imágenes claras de las caras de las personas que deseas reconocer.
    - Cada persona debe tener al menos una imagen, pero se recomienda tener varias imágenes en diferentes condiciones para mejorar la precisión.

2. **Procesamiento de las Imágenes:**
    - Carga cada imagen con OpenCV o dlib.
    - Convierte la imagen a escala de grises y detecta la cara.
    - Calcula el embedding facial utilizando el modelo de reconocimiento facial de dlib.
    - Agrega este embedding a la lista `face_embeddings`.

3. **Asociación con Nombres:**
    - Por cada embedding en `face_embeddings`, agrega el nombre correspondiente de la persona en la lista `names`.

Ajusta el umbral de coincidencia según sea necesario para obtener un rendimiento óptimo.

## Contribuciones

Las contribuciones son bienvenidas. Si encuentra algún error o desea agregar funcionalidades adicionales, no dude en crear un pull request o abrir un issue.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo `LICENSE` para más detalles.