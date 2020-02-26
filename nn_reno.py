#!/usr/bin/env python
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


data = pickle.load(open("Data/TRAININGDATA.SALAMP.xlsx K3-Part0.p", "rb"))
data2 = pickle.load(open("Data/TRAININGDATA.SALAMP.xlsx K3-Part1.p", "rb"))
df = data.append(data2)

# Change columns names of MIC in case there are similar column names
df.columns = [v for v in df.columns[:1]] + ["MIC_val"] + [v for v in df.columns[2:]]

dummies = pd.get_dummies(df.Antibiotic)
df = pd.concat([df, dummies], axis=1)

df = df.drop(columns=["Antibiotic"], axis=1)

df["MIC_val"] = df["MIC_val"].astype(float)
df["MIC_val"] = df["MIC_val"].astype(int)

# Get input and output columns
y = df.pop("MIC_val")
X = df

# total number of kmers in the dataset
kmers = len(X.columns)


dataset = tf.data.Dataset.from_tensor_slices((X.values, y.values))

# print out a couple samples
dataset = dataset.shuffle(len(X))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

size = len(y.values)
train_size = size - int(size*.1)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

val = val_dataset.enumerate()
val_ys = []
for _,v in val.as_numpy_iterator():
    _,y = v
    val_ys.append(y)

val_ys = np.array(val_ys)

# shuffle and batch the datasets
batch_size = 1
train_dataset = train_dataset.shuffle(len(X)).batch(batch_size)
val_dataset = val_dataset.shuffle(len(X)).batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(kmers/10, activation='relu'),
    tf.keras.layers.Dense(kmers/100, activation='relu'),
    tf.keras.layers.Dense(kmers/100, activation='relu'),
    tf.keras.layers.Dense(kmers/1000, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mean_squared_error',
    metrics=['mse'])

model.fit(train_dataset, epochs=1)

print('\n# Evaluate on test data')
results = model.evaluate(val_dataset)
print('test loss, test acc:', results)

pred = model.predict(val_dataset)
pred_fix = []
pred = np.array([v[0] for v in pred])

for i,v in enumerate(pred):
    t = round(v)
    if t > 6: t = 6
    if t < 0: t = 0
    pred[i] = t

r = abs(pred-val_ys)
r = r == 0
count = 0
for v in r:
    if v:
        count += 1
print("MIC acc: ", count/len(r))

r = abs(pred-val_ys)
r = r <= 1
count = 0
for v in r:
    if v:
        count += 1

print("MIC acc +-1: ", count/len(r))
