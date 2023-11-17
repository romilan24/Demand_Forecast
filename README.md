# CAISO Load Forecast

This repository aims to forecast the hourly system load of the California Independent System Operator (CAISO), which is essentially an aggregate of all electricity demand in California (~90% of electricity consumption).  The terms 'demand' and 'load' are used interchangeably.  Load is important because it allows both the system operator and market participants to optimize their operations and minimize the probability of a 'loss of load' event (blackout).

## Forecast Requirements

The goal of this project is to apply various machine learning algorithms to a forecasting problem. The challenge involves developing a forecasting library and using it to forecast the CAISO system load. The specific requirements are as follows:

1. Develop a forecasting library that can be used for various forecasting tasks.
2. Create models for forecasting, including Linear Regression, XGBoost, and Random Forest.
3. Provide methods for training models, generating forecasts, and evaluating performance.
4. Forecast the final 24 hours of the dataset where the target variable is null.
5. Compare the performance of different models using metrics and visualizations.

## Repository Structure

- `models.py`: Contains functions for training machine learning models.
- `utils.py`: Includes utility functions for data preprocessing, loading, and evaluation.
- `forecast.py`: Demonstrates how to use the library to develop models, generate forecasts, and compare their performance.
- `caiso_system_load.csv`: The dataset used for the forecasting task.
- `Actuals_20230731.csv`: Actual CAISO System Load for July 31st, 2023 downloaded from oasis.caiso.com
- `README.md`: This documentation file.

## How to Use

1. Clone this repository to your environment.

2. Ensure you have the required libraries installed:
pip install pandas numpy scikit-learn xgboost matplotlib

Python package versions used:
- pandas 1.5.3
- numpy 1.23.5
- XGBoost 1.7.3
- scikit-learn 1.2.0

3. Open and modify the paths in the `forecast.py` script to match your local file paths.

4. Run the `forecast.py` script to perform the forecast and model comparison.

## Assumptions

The code assumes that missing data in the dataset is interpolated and that the dataset contains relevant columns for the forecasting task.

## Results and Visualization

The code generates visualizations to compare the performance of different models. You can view the Mean Absolute Percentage Error (MAPE) for each model over the 24-hour Day Ahead interval and compare the forecasted values for each hour of the day.

Taking a look at the 'Model Predictions vs. Actual Load' graph we see that XGBoost and RandomForest (rf) models are less smooth than the linear model.  Interestingly the shape of the linear model was very good, but shifted down by an average of 2.5GWs across the day.  Assuming data isn't biased, we would want to investigate this a bit more closely to determine if we can leverage linear model in any way, else just blend XGBoost and rf models.

Observing the 'MAPE Comparison Between Models' chart, we note that during the morning Offpeak the XGBoost and rf models performed equally well (2% MAPE is very good) with the XGBoost model becoming divergent and volatile during midday.  Overall, we would reason that the Random Forest model performed the best per the given data.

## Improvements

If we were given more time to this assignment, we could very easily further improve upon these results by:
- including additional exogenous variables such as extra city temperatures (to better represent the CAISO service area), holiday flags, lags, and additional weather features such as humidity, wind speed, cloud cover, sunrise/sunset, etc.
- try ensemble methods such as including a stacking() function in the models.py to create a new model based on the sub-models or blending top 2 models together
- try ARIMA model w/ exogenous variables, cross-validation, etc.
- try hyperparameter tuning function to find optimal configuration for each model
- include Day-of Actuals up to the forecast hour, e.g. if forecast deadline is 10AM, then include data up until 9AM intra day



Enjoy forecasting with the CAISO Load Forecasting Library!
## Authors

- Diego Calderon

