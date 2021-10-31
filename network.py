from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('data/train/train.csv')
data_test = pd.read_csv('data/train/test.csv')

print(data_train.head())

labels = data_train[data_train.keys()[1:-1]]
pawpurality = data_train[data_train.keys()[-1]]

X_train = labels[:int(len(labels)*0.8)]
X_test = labels[int(len(labels)*0.8):]
y_train = pawpurality[:int(len(pawpurality)*0.8)]
y_test = pawpurality[int(len(pawpurality)*0.8):]


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),


    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(
    X_train,
    y_train,
    epochs=100, validation_data=(X_test, y_test), validation_freq=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)])


print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()


modelTree = RandomForestRegressor(n_estimators=100)
historyTree = modelTree.fit(X_train, y_train)


model1pred = model.predict(X_test)
model2pred = modelTree.predict(X_test)

pred1 = mean_absolute_error(y_test, model1pred)
pred2 = mean_absolute_error(y_test, model2pred)

print("Modelo 1: ", pred1)
print("Modelo 2: ", pred2)

#Get a random sample of the test data
print(model.predict(X_test[0:10]))
print(y_test[0:10])

print("Arbol de decisiones")
print(modelTree.predict(X_test[0:10]))
print(y_test[0:10])