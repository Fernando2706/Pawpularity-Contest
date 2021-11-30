import pandas as pd

train_data = pd.read_csv('data/train.csv')

id = train_data[train_data.keys()[:1]]
pawpurality = train_data[train_data.keys()[-1]]/100

X = []
y = []

import cv2
from tqdm import trange

SIZE_IMAGE=150

print('\nWe get the ID and popularity data\n')

for i in trange(len(id)):
    ID = id.iloc[i]
    pawp = pawpurality.iloc[i]
    path = 'data/train/'+ID.Id+'.jpg'
    image = cv2.imread(path)
    image = cv2.resize(image, (SIZE_IMAGE, SIZE_IMAGE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(SIZE_IMAGE, SIZE_IMAGE, 1)
    X.append(image)
    y.append(pawp)

import numpy as np

X = np.array(X).astype('float32')/255.0
y = np.array(y)

print('\nWe allocate 85% of the data to train and 15% to validate\n')

len_array = int(len(X)*0.85)

X_train = X[:len_array]
X_val = X[len_array:]
y_train = y[:len_array]
y_val = y[len_array:]

print(y_train)
print('\nWe are going to apply certain filters to the image\n')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5]
)

data_gen.fit(X)

import tensorflow as tf

print('\nWe create the CNN model\n')

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu',input_shape=(SIZE_IMAGE,SIZE_IMAGE,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2, 2))

model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2, 2))

model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2, 2))

model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2, 2))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512,activation='relu')),
model.add(tf.keras.layers.Dense(256,activation='relu')),
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
data_gen_train = data_gen.flow(X_train, y_train, batch_size=32)


from tensorflow.keras.callbacks import TensorBoard
tensorBoard = TensorBoard(log_dir='logs/photo_version/cnn_add')

history = model.fit(
    data_gen_train,
    epochs=50, batch_size=32,
    validation_data=(X_val, y_val),
    steps_per_epoch=int(np.ceil(len(X_train) / float(32))),
    validation_steps=int(np.ceil(len(X_val) / float(32))),
    callbacks=[tensorBoard]
)

model.save('data/photo_version/model.h5')

print('\nLet`s make a prediction\n')

import matplotlib.pyplot as plt

image_test = X_val[0]
#Reshape the image
image = image.reshape(1,150, 150,1)
#Plot the image
prediction = model.predict([image])
print(prediction)

