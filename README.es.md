# Concurso Kaggle
- 馃嚜馃嚫 Espa帽ol
- 馃嚭馃嚫 [English](https://github.com/Fernando2706/Pawpularity-Contest/blob/main/README.md)

> Como tecn贸logo, veo c贸mo la IA y la cuarta revoluci贸n industrial afectar谩n todos los aspectos de la vida de las personas.      Fei-Fei Li

En esta competencia, su tarea es predecir el compromiso con el perfil de una mascota en funci贸n de la fotograf铆a de ese perfil. Tambi茅n se le proporcionan metadatos etiquetados a mano para cada foto. Por lo tanto, el conjunto de datos para esta competencia comprende im谩genes y datos tabulares.

### Instalaci贸n
1. Instalar las librerias necesarias para el correcto funcionamiento del programa

    - Acceleracion mediante GPU (NVIDIA) y CUDA
        1. Instalar cuda & cudnn mediante el siguiente comando para *Arch Linux*

            ``` 
            sudo pacman -S cuda cudnn
            ```
        
        2. Instalar las librerias necesarias de python mediante el siguiente comando:

            ```
            pip install tensorflow tensorflow_datasets pandas numpy matplotlib opencv-python scipy scikit-learn 
            ```

        3. Correr el script `test_gpu.py` para saber si tenemos hardware compatible con tensorflow
    
    - Sin acceleracion mediante Cuda
         1. Instalar las librerias necesarias de python mediante el siguiente comando:

            ```
            pip install tensorflow tensorflow_datasets pandas numpy matplotlib opencv-python scipy scikit-learn 
            ```

Una vez tengamos todo instalado podremos proceder a correr el primer programa para detectar si se trata de un gato o un perro. Para ello usaremos el siguiente comando:
    ```python cnn_animals_detect.py```
