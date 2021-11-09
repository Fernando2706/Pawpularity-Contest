# Concurso Kaggle
- 🇪🇸 Español
- 🇺🇸 English

> Como tecnólogo, veo cómo la IA y la cuarta revolución industrial afectarán todos los aspectos de la vida de las personas.      Fei-Fei Li

En esta competencia, su tarea es predecir el compromiso con el perfil de una mascota en función de la fotografía de ese perfil. También se le proporcionan metadatos etiquetados a mano para cada foto. Por lo tanto, el conjunto de datos para esta competencia comprende imágenes y datos tabulares.

### Instalación
1. Instalar las librerias necesarias para el correcto funcionamiento del programa

    - Acceleracion mediante GPU (NVIDIA) y CUDA
        1. Instalar cuda & cudnn mediante el siguiente comando para *Arch Linux*

            ``` sudo pacman -S cuda cudnn ```
        
        2. Instalar las librerias necesarias de python mediante el siguiente comando:

            ```pip install tensorflow tensorflow_datasets pandas numpy matplotlib opencv-python scipy ```

        3. Correr el script `test_gpu.py` para saber si tenemos hardware compatible con tensorflow
    
    - Sin acceleracion mediante Cuda
         1. Instalar las librerias necesarias de python mediante el siguiente comando:

            ```pip install tensorflow tensorflow_datasets pandas numpy matplotlib opencv-python scipy ```

Una vez tengamos todo instalado podremos proceder a correr el primer programa para detectar si se trata de un gato o un perro. Para ello usaremos el siguiente comando:
    ```python cnn_animals_detect.py```