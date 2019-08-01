# -*- coding: utf-8 -*-
"""stacked_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L6fCAfRib4OpjpdJH7M3DzepsW6B0nJq
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

data_dim = 16
timesteps = 8
num_classes = 10

def get_data(normalized=0):
    # col_names = ['Date','Sunrise','Sunset','Daylength','HourofDay','HourNumber','Day','Day_No','RelativeFrequency','PercentageTripperHour','Numberoftripsperhour','Day_or_Night','1_or_0']
    # col_names = ['Date', 'Sunrise', 'Sunset', 'Daylength', 'HourofDay', 'HourNumber', 'Day_No.','NumberTripsperday']
    col_names = ['Date', 'Day_No','NumberTripsperday','PercentageTripperHour','1_or_0']
    stocks = pd.read_csv(r"Dataset_Daylight_Rainfall_Berlin4.csv", header=0)
    df = pd.DataFrame(stocks)

    print(df.columns)
    # df.drop(['Sunrise','Sunset','Daylength','HourofDay','HourNumber', 'Day','RelativeFrequency','Numberoftripsperhour','Day_or_Night'], axis=1, inplace=True)
    # df.drop([1,2,3,4,5,6,8,11])
    return df

#mount the drive

# from google.colab import drive
# drive.mount('/content/drive')

data = get_data()
labels = data.columns

data.drop(['Date'],inplace=True, axis='columns')

#data["1_or_0"] = data["1_or_0"].astype('category')

data["1_or_0"] = to_categorical(data["1_or_0"])

print (data.head())

train_size = int(len(data) * 0.95)
train, test = data[0:train_size], data[train_size:len(data)]
print('Observations: %d' % (len(data)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))

print (train.head())

x_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1:]

x_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1:]

print (x_train.head())
print (y_train.head())

data_dim =  x_train.shape[1] #number of input features
print(data_dim)

# # expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(1, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=64, epochs=5,
#           validation_data=(x_val, y_val))
# result = model.evaluate(x_val,y_val)

print(model.summary())

print (y_train.shape)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

trainX = np.array(x_train)
testX = np.array(x_test)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model.fit(trainX, y_train_cat,
          batch_size=64, epochs=100)
result = model.evaluate(testX,y_test_cat)

print('test: ', result)

print(model.predict(testX))

print (model.predict_classes(testX))

print (model.predict_proba(testX))