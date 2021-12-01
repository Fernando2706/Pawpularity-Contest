#Create an IA that will merge an CNN and a RNN
import os
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('data/train.csv',index_col='Id')
data_test = pd.read_csv('data/test.csv',index_col='Id')


img_size = 128
def create_CNN_model():
    input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    conv1 = input_layer
    conv1 = tf.keras.layers.Conv2D(32,(3, 3), activation='relu')(conv1)
    conv1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    conv2 = conv1
    conv2 = tf.keras.layers.Conv2D(64,(3, 3), activation='relu')(conv2)
    conv2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    conv3 = conv2
    conv3 = tf.keras.layers.Conv2D(128,(3, 3), activation='relu')(conv3)
    conv3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    conv4 = conv3
    conv4 = tf.keras.layers.Flatten()(conv4)
    conv4 = tf.keras.layers.Dense(512, activation='relu')(conv4)

    output_layer = conv4

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='CNN_model')
    return model

cnn_model = create_CNN_model()
cnn_model.summary()

def create_meta_model():
    input_layer = tf.keras.layers.Input(shape=(12,))
    ann = input_layer
    ann = tf.keras.layers.Dense(512, activation='relu')(ann)
    ann = tf.keras.layers.Dense(512, activation='relu')(ann)
    ann = tf.keras.layers.Dense(512, activation='relu')(ann)

    output_layer = ann
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='meta_model')
    return model

ann_model = create_meta_model()
ann_model.summary()   

def create_final_model(cnn_model, ann_model):
    layer = tf.keras.layers.Concatenate(axis=1)([cnn_model.output, ann_model.output])
    layer = tf.keras.layers.Dense(1, activation='relu')(layer)
    output_layer = layer
    model = tf.keras.models.Model(inputs=[cnn_model.input, ann_model.input], outputs=output_layer, name='final_model')
    return model

model = create_final_model(cnn_model, ann_model)
model.summary()

from sklearn.model_selection import train_test_split

data = data_train.sample(frac=1)

FEATURES = [
        'Subject Focus', 
        'Eyes', 
        'Face', 
        'Near', 
        'Action', 
        'Accessory', 
        'Group', 
        'Collage', 
        'Human', 
        'Occlusion', 
        'Info', 
        'Blur'
    ]

X_train,X_test,y_train,y_test = train_test_split(data[FEATURES], data['Pawpularity'], test_size=0.2, random_state=42)

from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image
def resize_image(image):
    path = os.path.join('data/train', '{}.jpg'.format(image))
    img = Image.open(path)
    img = img.resize((img_size, img_size))

    return img

class Data(Sequence):

    def __init__(self, X,target=None , batch_size=32):
        self.X = X
        self.target = target
        self.batch_size = batch_size
    
    def __len__(self):
        return int(len(self.X) / self.batch_size)
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx+1) * self.batch_size
        ids = self.X[start:end].index.values

        images = np.array([np.array(resize_image(id)) for id in ids])
        meta_data = np.array(self.X[start:end][FEATURES]).astype(np.float32)

        if self.target is not None:
            return [images,meta_data]

        target = np.array(self.target[start:end]).astype(np.float32)
        return [images,meta_data], target

train_data = Data(X_train, y_train)
print(train_data.__getitem__(0))
test_data = Data(X_test, y_test)

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001,
) , loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(train_data, epochs=50, validation_data=test_data,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])


