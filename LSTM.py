import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader as web
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler



sp500 = yf.download("^GSPC",start='2022-04-08',end='2023-04-06')

print(sp500)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(sp500["Close"].values.reshape(-1,1))

num_days = 30
x_train = []
y_train = []

for i in range(num_days, len(scaled_data)):
    x_train.append(scaled_data[i-num_days:i,0 ])
    y_train.append(scaled_data[i,0])

x_train= np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Now we build our model
model1 = Sequential()
#units is number of layers, a parameter to be experimented with
model1.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model1.add(Dropout(0.2))

model1.add(LSTM(units=50,return_sequences=True))
model1.add(Dropout(0.2))

model1.add(LSTM(units=50))
model1.add(Dropout(0.2))

#Predicts next closing price
model1.add(Dense(units=1))

model1.compile(optimizer='adam',loss='mean_squared_error')
model1.fit(x_train,y_train,epochs=25,batch_size=32)

#Now the model gets tested
test_start = dt.datetime(2022,4,8)
test_end = dt.datetime(2023,1,1)

sp500_test = yf.download("^GSPC",start=test_start,end=test_end)
actual_prices = sp500_test["Close"].values

total_dataset = pd.concat((sp500["Close"],sp500_test["Close"]),axis=0)

model_inputs = total_dataset[len(total_dataset) - len(sp500_test) - num_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Building test data
x_test = []
for i in range(num_days, len(model_inputs)):
    x_test.append(model_inputs[i-num_days:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_prices = model1.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)



#Prediction for next day closing price

# real_data = [model_inputs[len(model_inputs)+1 - num_days:len(model_inputs+1),0]]
# real_data = np.array(real_data)
# real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))
# prediction = model1.predict(real_data)
# prediction = scaler.inverse_transform(prediction)
# print(prediction)







