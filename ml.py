import pandas as pd
import numpy as np
import keras as k
from numpy import loadtxt


df = pd.read_csv("records.csv",low_memory=False)

print(df.head())

df["avg_mark"][df["avg_mark"]<=0] = np.nan
df["avg_mark"] = df["avg_mark"].replace(np.nan, round(df["avg_mark"].mean()))

df = df.dropna()
df = df.drop("id",axis = 1)

df.to_csv('clean_records.csv', index=False)

mydata = loadtxt("clean_records.csv", delimiter =",",skiprows = 1,usecols =(1,2,3,4,5,6,7))
x = mydata[:,0:6]
y = mydata[:,6]

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(20, input_dim=6, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x,y, epochs = 3000,batch_size = (df.shape[0]))

_,accuracy = model.evaluate(x, y)
print("Accuracy: ", accuracy)

