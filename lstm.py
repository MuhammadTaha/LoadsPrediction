from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_dim = 16
timesteps = 8
num_classes = 10

def get_data(normalized=0):

    # col_names = ['Date','Sunrise','Sunset','Daylength','HourofDay','HourNumber','Day','Day_No','RelativeFrequency','PercentageTripperHour','Numberoftripsperhour','Day_or_Night','1_or_0']
    # col_names = ['Date', 'Sunrise', 'Sunset', 'Daylength', 'HourofDay', 'HourNumber', 'Day_No.','NumberTripsperday']
    col_names = ['Date', 'Day_No','NumberTripsperday','PercentageTripperHour','1_or_0']
    stocks = pd.read_csv("Dataset_Daylight_Rainfall_Berlin4.csv", header=0)
    df = pd.DataFrame(stocks)

    print(df.columns)
    # df.drop(['Sunrise','Sunset','Daylength','HourofDay','HourNumber', 'Day','RelativeFrequency','Numberoftripsperhour','Day_or_Night'], axis=1, inplace=True)
    # df.drop([1,2,3,4,5,6,8,11])
    return df

data = get_data()
labels = data.columns
train ,test = train_test_split(data)

# print(train)
# print(test)


# # expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

# model.fit(x_train, y_train,
#           batch_size=64, epochs=5,
#           validation_data=(x_val, y_val))
# result = model.evaluate(x_val,y_val)

model.fit(train, labels,
          batch_size=64, epochs=5,
          validation_data=(test, labels))
result = model.evaluate(test,labels)

print(result)