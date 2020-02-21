#!/usr/bin/env python
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras.utils as util


# Retrieve data from pickles
table1 = pd.read_pickle("data/TRAININGDATA.SALAMP.xlsx K3-Part0.p")
table2 = pd.read_pickle("data/TRAININGDATA.SALAMP.xlsx K3-Part1.p")
table = table1.append(table2)

# Fix DataFrame column names
table.columns = [v for v in table.columns[:1]] + ["MIC_val"] + [v for v in table.columns[2:]]
dummies = pd.get_dummies(table.Antibiotic)
table = pd.concat([table, dummies], axis=1)
X = table.drop(table.columns[1], axis=1).drop("Antibiotic", axis=1)
y = table.iloc[:, 1]
#print("table:")
#print(table)
#print("X:")
#print(X)
#print("Y:")
#print(y)
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_test_up = pd.DataFrame([str(float(x) + 1) for x in y_test]).iloc[:, 0]
y_test_down = pd.DataFrame([str(float(x) - 1) for x in y_test]).iloc[:, 0]

#from tabulate import tabulate
#print(tabulate(X_train, headers='keys', tablefmt='psql'))
#print("X_train:")
#print(X_train)
print("X_train:")
print(X_train)
#print("y_train:")
#print(y_train)

X_train = X_train.values.reshape(3957, 9091, 1, 1)
X_test = X_test.values.reshape(1320, 9091, 1, 1)
print("After reshaping X_train:")
print(X_train)


quit()

# can't convert float to categorical so:
a_train = {}
classes_train = []
for item, i in zip(y_train, range(len(y_train))):
    a_train[str(i)] = item
    classes_train.append(str(i))

a_test = {}
classes_test = []
for item, i in zip(y_test, range(len(y_test))):
    a_test[str(i)] = item
    classes_test.append(str(i))


y_train = util.to_categorical(classes_train)
y_test = util.to_categorical(classes_test)
# print out a couple samples
#for feat, targ in dataset.take(5):
#  print ('Features: {}, Target: {}'.format(feat, targ))

kmers = len(X.columns)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(kmers, kernel_size = 1, activation='relu'),
    #tf.keras.layers.Conv2D(kmers/2, kernel_size = 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)

print('\n# Evaluate on test data')
results = model.evaluate(val_dataset)
print('test loss, test acc:', results)
