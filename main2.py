import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt2

def get_data(normalized=0):

    # col_names = ['Date','Sunrise','Sunset','Daylength','HourofDay','HourNumber','Day','Day_No','RelativeFrequency','PercentageTripperHour','Numberoftripsperhour','Day_or_Night','1_or_0']
    col_names = ['Date', 'Sunrise', 'Sunset', 'Daylength', 'HourofDay', 'HourNumber', 'Day_No','NumberTripsperday']
    stocks = pd.read_csv("Dataset_Daylight_Rainfall_Berlin2.csv", header=0, names=col_names)
    df = pd.DataFrame(stocks)
    # df.drop(df.columns[[0,3,5,6]], axis=1, inplace=True)
    return df

df = get_data(0)
df.tail()


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    # print(stock)
    # sequence_length = seq_len + 1
    # result = []
    # for index in range(len(data) - sequence_length):
    #     result.append(data[index: index + sequence_length])

    result = np.array(data)
    # print(result)
    row = round(0.8 * result.shape[0])
    # print(row)
    train = result[:int(row), :]
    print("dfsdfd")
    print(len(train))
    x_train = train[:, :-1]
    print(x_train)
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]

# def build_model(layers):
#     model = Sequential()
#
#     model.add(LSTM(
#         input_dim=layers[0],
#         output_dim=layers[1],
#         return_sequences=True))
#     model.add(Dropout(0.2))
#
#     model.add(LSTM(
#         layers[2],
#         return_sequences=False))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(
#         output_dim=layers[2]))
#     model.add(Activation("linear"))
#
#     start = time.time()
#     model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
#     print("Compilation Time : ", time.time() - start)
#     return model

def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))
        model.add(Dense(1,init='uniform',activation='relu'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model


window = 5
X_train, y_train, X_test, y_test = load_data(df[::-1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

model = build_model2([3,window,1])

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


p = model.predict(X_test)
# for u in range(len(y_test)):
#     pr = p[u][0]
#     ratio.append((y_test[u]/pr)-1)
#     diff.append(abs(y_test[u]- pr))
    #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))




plt2.plot(p,color='red', label='prediction')
plt2.plot(y_test,color='blue', label='y_test')
plt2.legend(loc='upper left')
plt2.show()