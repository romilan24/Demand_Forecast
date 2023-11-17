import pandas as pd
import numpy as np
from models import train_model
from utils import load_data, rename_columns, split_data, calculate_mape
from datetime import datetime
import matplotlib.pyplot as plt

# Assuming you have a list of holidays
holidays = [datetime(2020, 12, 25), datetime(2021, 1, 1)]  # Add more holidays

# Initialize models
models = ["LinearModel", "XGBoostModel", "rfModel"]

#rename column headers
column_mapping = {
    'interval_start_time': 'datetime',
    'CAISO_system_load': 'load',
    'temp_forecast_dayahead_bakersfield': 'BF_temp',
    'temp_forecast_dayahead_los_angeles': 'LA_temp',
    'temp_forecast_dayahead_san_francisco': 'SF_Temp',
    'dewpoint_forecast_dayahead_bakersfield': 'BF_dewpoint',
    'dewpoint_forecast_dayahead_los_angeles': 'LA_dewpoint',
    'dewpoint_forecast_dayahead_san_francisco': 'SF_dewpoint'
}

# Specify the date ranges
train_start_date = pd.to_datetime('2020-01-01').date()
train_end_date = pd.to_datetime('2023-07-30').date()
predict_date = pd.to_datetime('2023-07-31').date()

#library, change to file path where .csv files located
path = 'C:/Users//'

# Loads the data, creates date and hour columns, creates lags, and holiday col
data = load_data(path + 'caiso_system_load.csv')

#renames the column headers
data = rename_columns(data, column_mapping)

#filter train/test data for start/end dates
df = data[(data['date'] >= train_start_date) & (data['date'] <= train_end_date)]

# Split data
X_train, X_test, y_train, y_test = split_data(df)

# Initialize dictionaries to store forecasts
hourly_forecasts = {model_name: [] for model_name in models}

##Model fit##
# Loop through models and hours for training
for model_name in models:
    for hour in range(1, 25):
        
        X_train_hour = X_train[X_train['he'] == hour].drop(['he', 'date'], axis=1)
        y_train_hour = y_train[X_train['he'] == hour]
        
        model = train_model(model_name, X_train_hour, y_train_hour)
        
        # Assuming you have 2023-07-31 data for exogenous features
        X_forecast = data[data['date'] == pd.to_datetime('2023-07-31').date()]
        X_forecast = X_forecast[X_forecast['he'] == hour].drop(['datetime', 'load', 'date', 'he'], axis=1)
        y_forecast = model.predict(X_forecast)
        
        #print(f"Model: {model_name}, Hour: {hour}, Forecasted Load: {y_forecast}")
        
        hourly_forecasts[model_name].append(y_forecast)

##Compare Performance##
# Import actual data from 'Actuals_20230731.csv'; pulled from oasis.caiso.com
actual_data = pd.read_csv(path + 'Actuals_20230731.csv', usecols=['OPR_DT', 'OPR_HR', 'MW'])

# Plot the three model predictions along with the actual load
plt.figure(figsize=(14, 6))
for model_name in models:
    forecasts = []
    for hour in range(1, 25):
        forecast_values = hourly_forecasts[model_name][hour - 1]
        forecasts.append(forecast_values)

    plt.plot(range(1, 25), forecasts, marker='o', label=model_name)

# Extract actual load data
actual_load = actual_data['MW']

# Plot the actual load in red
plt.plot(range(1, 25), actual_load, color='red', marker='o', label='Actual Load')

plt.title("Model Predictions vs. Actual Load")
plt.xlabel("Hour")
plt.ylabel("Load")
plt.legend(loc="upper left")
plt.show()

# Plot the MAPE values for the selected models
plt.figure(figsize=(14, 6))
for model_name in models:
    mape_values = []
    for hour in range(1, 25):
        y_true_hour = actual_data.loc[actual_data['OPR_HR'] == hour, 'MW']
        mape = calculate_mape(y_true_hour, hourly_forecasts[model_name][hour - 1])
        mape_values.append(mape)
    
    plt.plot(range(1, 25), mape_values, marker='o', label=model_name)

plt.title("MAPE Comparison Between Models")
plt.xlabel("Hour")
plt.ylabel("MAPE")
plt.legend(loc="upper left")
plt.show()
