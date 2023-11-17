import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    
    #if missing data, interpolate
    if df.isnull().sum().any():
        print('Missing data detected. Interpolating missing values using linear interpolation.')
        df = df.interpolate(method='linear')
    
    zone = 'America/Los_Angeles'
    
    #create date and hour ending columns
    df['date'] = pd.to_datetime(df['interval_start_time'].str.strip(), utc=True).dt.tz_convert(zone)
    df['he'] = pd.to_datetime(df['interval_start_time'].str.strip(), utc=True).dt.tz_convert(zone)

    df['date'] = pd.to_datetime(df['date'], format= '%Y-%m-%d')

    df['date'] = df['date'].dt.date
    df['he'] = df['he'].dt.hour + 1
    
    return df

def rename_columns(df, column_mapping):
    # Use the rename method to rename the columns
    df = df.rename(columns=column_mapping)
    return df


'''
    # Create lag features
    for i in range(1, 8):
        df['lag_' + str(i)] = df['CAISO_Load'].shift(i)

    # Create holiday flags
    df['is_holiday'] = df['datetime'].dt.is_holiday
'''

def split_data(df, test_size=0.2, random_state=35):

    X = df.drop(['load', 'datetime'], axis=1)
    y = df['load']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100