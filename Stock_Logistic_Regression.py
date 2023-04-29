import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

sp500 = yf.download("^GSPC",start='2022-04-08',end='2023-04-06')
data = sp500['Adj Close'].pct_change()*100
data=data.rename('Current')
data=data.reset_index()
#creating lag to access prior prices
for i in range(1,11):
    data['Lag '+str(i)]=data['Current'].shift(i)
data=data.dropna()
#Creating a column to represent if price moved up or down. 1 represents price moving up and 0 represents down.
data['Movement'] = [1 if i>0 else 0 for i in data['Current']]
#add column with all 1's
data = sm.add_constant(data)
x = data[['const','Lag 1','Lag 2','Lag 3','Lag 4','Lag 5','Lag 6','Lag 7','Lag 8','Lag 9','Lag 10']]
y=data.Movement
model1=sm.Logit(y,x)
outcome=model1.fit()
print(outcome.summary())
prediction = outcome.predict(x)



# x_train = data[data.Date.dt.year < 2023 ][['const','Lag 1','Lag 2','Lag 3']]
# y_train = data[data.Date.dt.year < 2023 ][['Movement']]
# x_test = data[data.Date.dt.year == 2023] [['const','Lag 1','Lag 2','Lag 3']]
# y_test = data[data.Date.dt.year == 2023 ][['Movement']]
#
# model1 = sm.Logit(y_train,x_train)
# outcome = model1.fit()
# prediction = outcome.predict(x_test)



