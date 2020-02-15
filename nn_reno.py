#!/usr/bin/env python
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


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
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

size = len(y.values)
train_size = size - int(size*.1)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# shuffle and batch the datasets
batch_size = 10
train_dataset = train_dataset.shuffle(len(X)).batch(batch_size)
val_dataset = val_dataset.shuffle(len(X)).batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(kmers, activation='relu'),
    tf.keras.layers.Dense(kmers, activation='relu'),
    tf.keras.layers.Dense(kmers/2, activation='relu'),
    tf.keras.layers.Dense(kmers/2, activation='relu'),
    tf.keras.layers.Dense(kmers/8, activation='relu'),
    tf.keras.layers.Dense(kmers/100, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(train_dataset, epochs=2)

print('\n# Evaluate on test data')
results = model.evaluate(val_dataset)
print('test loss, test acc:', results)
