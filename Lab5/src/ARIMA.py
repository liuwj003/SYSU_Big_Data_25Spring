import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from tqdm import tqdm

def load_and_prepare_data(data_path):
    """
    Load and prepare stock data from csv file, for ARIMA model.
    """
    df = pd.read_csv(data_path)
    df['Close'] = df['Close'].ffill()
    stock_data = np.array(df['Close'])

    return stock_data


def test_stationarity(timeseries):
    """
    Test stationarity of a time series using the 
    Augmented Dickey-Fuller test.

    :param timeseries: 1-d array, time series data
    :return: bool, True if stationary, False otherwise
    """
    # remove NaN, and perform ADF test
    adf_stat, p_value, *_ = adfuller(timeseries.dropna())
    print("ADF Test Statistic(检验统计量): ", adf_stat)
    print("P-value(显著性水平): ", p_value)

    if p_value <= 0.05:
        print("The timeseries is stationary.")
        return True
    else:
        print("The timeseries is non-stationary.")
        return False

def make_stationary(timeseries):
    """
    Make a time series stationary by differencing

    :param timeseries: 1-d array, time series data
    :return: differenced_series: 1-d array, differenced time series
    :return: d: int, differencing order
    """
    # calculate y(t) - y(t-1)
    diff_timeseries = timeseries.diff().dropna()

    if test_stationarity(diff_timeseries):
        return diff_timeseries, 1
    else:
        second_diff_timeseries = diff_timeseries.diff().dropna()
        return second_diff_timeseries, 2


def plot_acf_pacf(timeseries):
    """
    Plot ACF and PACF of a time series

    :param timeseries: 1-d array, time series data
    """
    plt.figure(figsize=(12, 6))
    
    # Plot ACF
    plt.subplot(121)
    plot_acf(timeseries, ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function')
    
    # Plot PACF
    plt.subplot(122)
    plot_pacf(timeseries, ax=plt.gca(), lags=40)
    plt.title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.savefig('acf_pacf.png')
    plt.show()

def train_and_forecast_arima(train_series, test_series, p, d, q, mean, std):
    """
    Train ARIMA model and make predictions with rolling forecast (normalized data).

    :param train_series: normalized training Series
    :param test_series: normalized testing Series
    :param mean: original mean, for denormalizing predictions
    :param std: original std, for denormalizing predictions
    :return: denormalized predictions, last fitted model
    """
    history = list(train_series)
    predictions = []

    for t in tqdm(range(len(test_series))):
        model = ARIMA(history, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        pred = model_fit.forecast()[0]
        predictions.append(pred)
        history.append(test_series.iloc[t])

    # Denormalize predictions before returning
    denorm_predictions = denormalize(predictions, mean, std)
    return denorm_predictions, model_fit

def multi_step_forecast_arima(train_series, steps, p, d, q, mean, std):
    """
    Fit ARIMA once and forecast next N steps.

    :param train_series: normalized training data
    :param steps: number of steps to forecast
    :param mean: original mean for denormalization
    :param std: original std for denormalization
    :return: denormalized predictions, fitted model
    """
    model = ARIMA(train_series, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=steps)

    denorm_predictions = denormalize(predictions, mean, std)
    return denorm_predictions, model_fit

def evaluate_model(y_true, y_pred):
    """
    Evaluate the ARIMA model performance with MAE and RMSE

    :param y_true: 1-d array, true values
    :param y_pred: 1-d array, predicted values
    :return: mae: float, mean absolute error
    :return: rmse: float, root mean squared error
    """
    mae = mean_absolute_error(y_true, y_pred)   
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    
    return mae, rmse

def normalize_series(series):
    """
    Normalize time series using z-score (standard score).
    :return: normalized_series, mean, std
    """
    mean = series.mean()
    std = series.std()
    normalized = (series - mean) / std
    return normalized, mean, std


def denormalize(predictions, mean, std):
    """
    Convert normalized predictions back to original scale.
    """
    return [y * std + mean for y in predictions]


def main():
    # Load and prepare data
    data_path = 'Microsoft_stock_history.csv'
    stock_data = load_and_prepare_data(data_path)
    
    # Convert to pandas Series
    time_series = pd.Series(stock_data)
    
    # Test stationarity
    is_stationary = test_stationarity(time_series)
    
    # If not stationary, make it stationary
    if not is_stationary:
        stationary_series, d = make_stationary(time_series)
        print(f"Differencing order (d): {d}")
    else:
        stationary_series, d = time_series, 0
    
    # Plot ACF and PACF to help identify p and q values
    plot_acf_pacf(stationary_series)
    
    # Based on ACF and PACF visualization, set appropriate p and q value
    p, q = 1, 1
    
    # Normalize the full series
    normalized_series, mean, std = normalize_series(time_series)
    
    predict_seven_days = True

    if not predict_seven_days:
        # Split data
        test_ratio = 0.2
        # test_ratio = 7/8956
        train_size = int(len(stock_data) * (1 - test_ratio))

        # train_series = stock_data[:train_size]
        # test_series = stock_data[train_size:]
        # Use normalized data for training and testing
        train_series = normalized_series[:train_size]
        test_series = normalized_series[train_size:]

    
        # Train ARIMA model and make rolling predictions
        predictions, model_fit = train_and_forecast_arima(train_series, test_series, p, d, q, mean, std)
        print(model_fit.summary())

        # Evaluate model performance
        test_series_denorm = denormalize(test_series, mean, std)
        mae, rmse = evaluate_model(test_series_denorm, predictions)
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # Plot actual vs predicted values
        plt.figure(figsize=(12, 6))
        plt.plot(time_series, label='Actual Data')

        # use test_series.index to align the predictions with the actual data
        plt.plot(test_series.index, predictions, color='red', label='Predictions')
        plt.axvline(x=test_series.index[0], color='green', linestyle='--', label='Train/Test Split')
        plt.title(f'ARIMA({p},{d},{q}) Forecast - MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ARIMA_stock_prediction.png')
        plt.show()
    else:
        forecast_steps = 7
        train_series = normalized_series[: -forecast_steps]
        test_series = normalized_series[-forecast_steps:]

        predictions, model_fit = multi_step_forecast_arima(train_series, forecast_steps, p, d, q, mean, std)
        print(model_fit.summary())

        test_series_denorm = denormalize(test_series, mean, std)

        mae, rmse = evaluate_model(test_series_denorm, predictions)
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        plt.figure(figsize=(12, 6))

        last_7_days_actual = time_series[-forecast_steps:]
        x_actual = range(len(time_series) - forecast_steps, len(time_series))

        x_pred = x_actual
        plt.plot(x_actual, last_7_days_actual, label='Actual Data', color='blue')

        plt.plot(x_pred, predictions, color='red', label='7-day Forecast')

        plt.axvline(x=x_actual[0], color='green', linestyle='--', label='Train/Test Split')

        plt.title(f'ARIMA({p},{d},{q}) 7-step Forecast - MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ARIMA_7day_forecast.png')
        plt.show()


if __name__ == "__main__":
    main()