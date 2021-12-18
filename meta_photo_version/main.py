#Create an IA that will merge an CNN and a RNN
print('Importing packages...')
import os
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

print('Loading data...')
data_train = pd.read_csv('data/train.csv',index_col='Id')
data_test = pd.read_csv('data/test.csv',index_col='Id')


img_size = 128
print('Creating CNN model...')
def create_CNN_model():
    input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    conv1 = input_layer
    conv1 = tf.keras.layers.Conv2D(128,(3, 3), activation='relu')(conv1)
    conv1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    conv2 = conv1
    conv2 = tf.keras.layers.Conv2D(128,(3, 3), activation='relu')(conv2)
    conv2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    conv3 = conv2
    conv3 = tf.keras.layers.Conv2D(128,(3, 3), activation='relu')(conv3)
    conv3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    conv4 = conv3
    conv4 = tf.keras.layers.Conv2D(128,(3, 3), activation='relu')(conv4)
    conv4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    conv5 = conv4
    conv5 = tf.keras.layers.Conv2D(128,(3, 3), activation='relu')(conv5)
    conv5 = tf.keras.layers.MaxPooling2D((2, 2))(conv5)
    conv6 = conv5
    conv6 = tf.keras.layers.Flatten()(conv6)
    conv6 = tf.keras.layers.Dense(512, activation='relu')(conv6)

    output_layer = conv6

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='CNN_model')
    return model

cnn_model = create_CNN_model()
cnn_model.summary()
print('Creating ANN model...')
def create_meta_model():
    input_layer = tf.keras.layers.Input(shape=(12,))
    ann = input_layer
    ann = tf.keras.layers.Dense(512, activation='relu')(ann)
    ann = tf.keras.layers.Dense(512, activation='relu')(ann)
    ann = tf.keras.layers.Dense(256, activation='relu')(ann)
    ann = tf.keras.layers.Dense(256, activation='relu')(ann)

    output_layer = ann
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='meta_model')
    return model

ann_model = create_meta_model()
ann_model.summary()   
print('Creating final model...')
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
def resize_image(image, path='data/train'):
    path = os.path.join(path, '{}.jpg'.format(image))
    img = Image.open(path)
    img = img.resize((img_size, img_size))

    return img
    
import math 

class Data(Sequence):

    def __init__(
        self, 
        x, 
        target=None, 
        batch_size=32,
        path='data/train'
    ):
        self.x = x
        self.target = target
        self.path = path
        self.batch_size = batch_size
    
    def __len__(self):
        # Return the number of batches per epoch
        return math.ceil(len(self.x) / self.batch_size)
    
    def __getitem__(self, idx):
        # Return one batch of data
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        ids = self.x[start_idx : end_idx].index.values
        
        images = np.array([np.array(resize_image(id,self.path )) for id in ids])
        meta = np.array(self.x[start_idx : end_idx][FEATURES]).astype(np.float32)
        
        if self.target is None or not self.target.any():
            return [images, meta]
        
        target = np.array(self.target[start_idx : end_idx]).astype(np.float32)
        return [images, meta], target

train_data = Data(X_train, y_train)
test_data = Data(X_test, y_test)

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001,
) , loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
steps_per_epoch = math.ceil(len(train_data) / 32)
history = model.fit(
    train_data,
    steps_per_epoch=100,
    epochs=50,
    validation_data=test_data,
    validation_steps=math.ceil(len(test_data) / 32),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ]
)
model.save('data/meta_photo/model.h5')


validate_data = Data(data_test,path='data/test')
print(validate_data.__len__() )
y_pred = model.predict(validate_data.__getitem__(0))

path_to_image = '4e429cead1848a298432a0acad014c9d'
meta_data = [ 0,0,0,1,0,1,1,1,0,1,1,1 ]

prediction = model.predict([np.array([np.array(resize_image(path_to_image,path='data/test' ))]), np.array(meta_data)])