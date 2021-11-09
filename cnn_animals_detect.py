# TODO: Crear una red neuronal para detectar si una foto es un gato o un perro, para despues procesarla correctamente
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from os import system
# Limpiamos la consola para mejor comodidad al desarollar
system('clear')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data, metadata = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)
print(metadata)

TAMANO_IMG = 100

data_train = []

SIZE_IMAGE = 150

for i, (image, label) in enumerate(data['train']):
    image = cv2.resize(image.numpy(), (SIZE_IMAGE, SIZE_IMAGE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(SIZE_IMAGE, SIZE_IMAGE, 1)
    data_train.append([image, label])

X = []
y = []

for (img, label) in data_train:
    X.append(img)
    y.append(label)

X = np.array(X).astype('float32')/255.0
y = np.array(y)

print(y, X.shape)

len_array = int(len(X)*0.85)
print(len_array)
X_train = X[:len_array]
X_val = X[len_array:]
y_train = y[:len_array]
y_val = y[len_array:]

# Vamos a aumentar los datos disponibles

data_gen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=20,
    zoom_range=[0.8, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)


data_gen.fit(X)

model_CNN_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(SIZE_IMAGE, SIZE_IMAGE, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),


    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_CNN_AD.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])


data_gen_train = data_gen.flow(X_train, y_train, batch_size=32)
tensorBoardCnn2 = TensorBoard(log_dir='logs/cnn_add')


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stop = EarlyStopping(patience=10)
learning_rr = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [tensorBoardCnn2, early_stop, learning_rr]

history = model_CNN_AD.fit(
    data_gen_train,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val),
    steps_per_epoch=int(np.ceil(len(X_train) / float(32))),
    validation_steps=int(np.ceil(len(X_val) / float(32))),
    callbacks=callbacks
)

model_CNN_AD.save('data/cnn_detect_animal.h5')


predictions = model_CNN_AD.predict(X_val)
acc = model_CNN_AD.evaluate(X_val, y_val)

for i in range(0, 5):
    image = X_val[i]
    plt.imshow(image.reshape(SIZE_IMAGE, SIZE_IMAGE), cmap='gray')
    #Reshape the image
    image = image.reshape(1,150, 150,1)
    #Plot the image
    prediction = model_CNN_AD.predict([image])
    if prediction[0][0] > 0.5:
        print('Perro')
    else:
        print('Gato')
    plt.show()

