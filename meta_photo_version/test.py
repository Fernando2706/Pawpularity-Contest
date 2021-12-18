import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image

#Clear console
os.system('clear')
print("Tensorflow version: ", tf.__version__)
print("Numpy version: ", np.__version__)
print("OpenCV version: ", cv2.__version__)
print( 'Loading the model...' )
model = tf.keras.models.load_model('data/meta_photo/model.h5')
print( 'Model loaded.' )

test_data = pd.read_csv('data/test.csv')
predictions = []
for i in range(len(test_data)):
    path = os.path.join('data/test/', '{}.jpg'.format(test_data['Id'][i]))
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)
    meta_data = test_data.iloc[ i, 1: ].values.astype(np.float32)
    print( 'Predicting for image {}...'.format(i) )
    prediction = model.predict([
        np.array([img]),
        np.array([meta_data])
    ])
    print( 'Prediction: {}'.format(prediction) )
    print( 'Done.' )
    print( 'Saving the prediction...' )
    predictions.append((test_data['Id'][i] , prediction[0][0]))

predictions = pd.DataFrame(predictions, columns=['Id', 'Pawpularity'])
predictions.to_csv('data/meta_photo/predictions.csv', index=False)
print( 'Predictions submitted.' )


