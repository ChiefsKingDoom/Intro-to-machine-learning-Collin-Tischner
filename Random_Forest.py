import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

sp500 = yf.download("^GSPC",start='2022-04-08',end='2023-04-06')
sp500.index = pd.to_datetime(sp500.index)
print(sp500)
#Creating a metric for directional movement
#Shifts all prices back one day
sp500["Tomorrow"] = sp500["Close"].shift(-1)
#Sets target (as integer rather than boolean)
sp500["Target"]=(sp500["Tomorrow"] > sp500["Close"]).astype(int)

model1=RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=999)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close","Volume","Open","High","Low"]
model1.fit(train[predictors],train["Target"])
predictions = model1.predict(test[predictors])
predictions = pd.Series(predictions)
print(accuracy_score(test["Target"],predictions))
