#!/usr/bin/env python
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras.utils as util
import numpy as np
import sklearn
import os

filename = '.model.cnn'

# Retrieve data from pickles
table1 = pd.read_pickle("data/TRAININGDATA.SALAMP.xlsx K3-Part0.p")
table2 = pd.read_pickle("data/TRAININGDATA.SALAMP.xlsx K3-Part1.p")
table = table1.append(table2)

# Fix DataFrame column names
table.columns = [v for v in table.columns[:1]] + ["MIC_val"] + [v for v in table.columns[2:]]
dummies = pd.get_dummies(table.Antibiotic)
table = pd.concat([table, dummies], axis=1)
X_ = table.drop(table.columns[1], axis=1).drop("Antibiotic", axis=1)
# need to delete one more column for easy factorization
X = X_.drop(X_.columns[9090], axis=1).drop("AMC", axis=1)

y = table.iloc[:, 1]
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("y_train:\n", y_train)
#print("y_test:\n", y_test)

X_train = X_train.values.reshape(3535,  71, 128, 1)
X_test = X_test.values.reshape(1742,  71, 128, 1)

if os.path.isfile(filename):
    model = tf.keras.models.load_model(filename)
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(100, kernel_size = [35, 64], activation='tanh', input_shape=(71, 128, 1)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(50, kernel_size = [16, 32], activation='tanh'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(25, kernel_size = [8, 16], activation='tanh'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(12, kernel_size = [4, 8], activation='tanh'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
        ])

    model.compile(optimizer='adam',
        loss=tf.keras.losses.mean_squared_error,
        metrics=['mse'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2) 
    model.save(filename)
#end of if 

model.summary()

#model.evaluate(X_test, y_test)

predictions = model.predict_classes(X_test)

correct = 0
y = pd.to_numeric(y_test)
for i in range(len(predictions)):
    if abs(predictions[i] - y.values[i]) <= 1:
        correct += 1

print("2 neighbour dilution Accuracy: ", 100*correct/i)

conf_mat = sklearn.metrics.confusion_matrix(y.values, predictions)
print(conf_mat)
