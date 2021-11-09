# Kaggle Contest
- 🇪🇸 [Spanish](https://github.com/Fernando2706/Pawpularity-Contest/blob/main/README.es.md)
- 🇺🇸 English

> As a technologist, I see how AI and the fourth industrial revolution will affect all aspects of people's lives. Fei-Fei Li

In this competition, your task is to predict engagement with a pet's profile based on the photograph of that profile. You are also provided with hand-tagged metadata for each photo. Therefore, the data set for this competition comprises images and tabular data.

### Installation
1. Install the necessary libraries for the correct operation of the program

    - Acceleration through GPU (NVIDIA) and CUDA
        1. Install cuda & cudnn using the following command for *Arch Linux*

            ```
            sudo pacman -S cuda cudnn
            ```
        
        2. Install the necessary python libraries using the following command:

            ``` 
            pip install tensorflow tensorflow_datasets pandas numpy matplotlib opencv-python scipy scikit-learn 
            ```

        3. Run the `test_gpu.py` script to see if we have hardware compatible with tensorflow
    
    - No acceleration through Cuda
         1. Install the necessary python libraries using the following command:

            ``` 
            pip install tensorflow tensorflow_datasets pandas numpy matplotlib opencv-python scipy scikit-learn
             ```

Once we have everything installed we can proceed to run the first program to detect if it is a cat or a dog. For this we will use the following command:
    ```python cnn_animals_detect.py```
