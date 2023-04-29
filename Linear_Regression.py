import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import yfinance as yf
import statsmodels.api as sm


sp500 = yf.download("^GSPC",start='2022-04-08',end='2023-04-06')
data = sp500['Adj Close'].pct_change()*100
data=data.rename('Current')
data=data.reset_index()
#creating lag to access prior prices
for i in range(1,4):
    data['Lag '+str(i)]=data['Current'].shift(i)
data=data.dropna()
#Creating a column to represent if price moved up or down. 1 represents price moving up and 0 represents down.
data['Movement'] = [1 if i>0 else 0 for i in data['Current']]
#add column with all 1's
data = sm.add_constant(data)
x = data[['const','Lag 1','Lag 2','Lag 3']]
y=data.Movement
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
model1 = linear_model.LinearRegression()
model1.fit(x_train,y_train)
y_prediction = model1.predict(x_test)
accuracy = model1.score(x_test,y_test)
print(accuracy)
print(r2_score(y_test,y_prediction))









# #sp500 = pd.read_csv('/Users/collintischner/OneDrive/Documents/sp500.csv')
# sp500 = yf.download("^GSPC",start='2022-04-08',end='2023-04-06')
# x = sp500.iloc[:,2].values
# y = sp500.iloc[:,4].values
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
# model1 = linear_model.LinearRegression()
# model1.fit(x_train,y_train)
# y_prediction = model1.predict(x_test)
# accuracy = model1.score(x_test,y_test)
# print(accuracy)




# import numpy as np
# import yfinance as yf
# from sklearn.linear_model import LinearRegression

# sp500 = yf.download("^GSPC",start='2022-04-08',end='2023-04-06')
# #Taking log of returns to normalize data
# sp500['Returns'] = np.log(sp500.Close.pct_change()+1)
#
# #assigning lags and their names. The lags are used to get assigned coefficients in the regression model
# def lag(data, nlags):
#     lags = []
#     for i in range(1,len(lags)+1):
#         data['Lag '+str(i)] = data['returns'].shift(i)
#         lags.append('Lag '+str(i))
#     return lags
# #making 3 lags
# sp500['lags'] = lag(sp500,3)
# sp500.dropna()
# model1 = LinearRegression()
# model1.fit(sp500['lags'],sp500['Returns'])
# sp500['Prediction'] = model1.predict(sp500['lags'])
# #sp500['Direction'] = [1 if i>0 else -1 for i in sp500.Prediction]



