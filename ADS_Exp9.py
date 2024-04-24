import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

df=pd.read_csv('ads.csv',index_col='Date' ,parse_dates=True)
df=df.dropna()
print('Shape of data',df.shape)
df.head()
df

df['Consumption'].plot(figsize=(15,5))

def ad_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
ad_test(df['Consumption'])

print(df.shape)
train=df.iloc[:-40]
test=df.iloc[-40:]
print(train.shape,test.shape)

# Fit SARIMA model using auto_arima
stepwise_fit = auto_arima(train['Consumption'], seasonal=True, m=10, trace=True, suppress_warnings=True)

# Get the best SARIMA model order parameters
best_order = stepwise_fit.order
best_seasonal_order = stepwise_fit.seasonal_order

print("Best SARIMA model order parameters (p, d, q):", best_order)
print("Best SARIMA seasonal order parameters (P, D, Q, s):", best_seasonal_order)

# Define SARIMA model parameters
order = best_order  # Non-seasonal order (p, d, q)
seasonal_order = best_seasonal_order # Seasonal order (P, D, Q, s)

# Fit SARIMA model
sarima_model = SARIMAX(train['Consumption'], order=order, seasonal_order=seasonal_order)
sarima_model_fit = sarima_model.fit()

# Print SARIMA model summary
print(sarima_model_fit.summary())

# Specify start and end points for predictions
start = len(train)  # Start index for predictions
end = len(train) + len(test) - 1  # End index for predictions

sarima_model = SARIMAX(train['Consumption'] , order = best_order, seasonal_order = best_seasonal_order)
sarima_model_fit = sarima_model.fit()

# Generate SARIMA predictions
sarima_pred = sarima_model_fit.predict(start=start, end=end, typ='levels').rename('SARIMA Predictions')

# Set index of predictions to match the dates in the test dataset
sarima_pred.index = test.index

# Set the figure size
plt.figure(figsize=(15, 6))  # Adjust the width and height as needed

# Plot SARIMA predictions and test data
sarima_pred.plot(legend=True)
test['Consumption'].plot(legend=True)

# Show the plot
plt.show()