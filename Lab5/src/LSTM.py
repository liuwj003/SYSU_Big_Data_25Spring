import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting
    nn.LSTM see: # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # input: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(input)
        # lstm_out: (batch_size, seq_len, hidden_size)
        # and we only want the last time step output
        out = self.fc(lstm_out[:, -1, :])

        return out
    
def train(model, train_loader, epoch, criterion, optimizer):
    """
    Train the LSTM model.

    :param model: LSTM model
    :param train_loader: DataLoader
    :param epoch: int
    :param criterion: loss function
    :param optimizer: optimizer
    """
    model.train()
    batch_loss_list = []

    for idx, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_loss_list.append(loss.item())
        
        if idx % 50 == 0:
            print(f"Epoch:{epoch}, Batch:{idx}, Loss:{loss.item():.8f}")

def load_and_prepare_data(data_path):
    """
    Load and prepare stock data from csv file, for LSTM model. 

    :param data_path: path to the data file
    :return scaled_close: 2-d array, like: (days, 1), scaled close price
    :return scaler: MinMaxScaler
    """
    # load data 
    df = pd.read_csv(data_path)
    # select close price, and fill missing value
    df['Close'] = df['Close'].ffill()
    close_prices = df[['Close']].values  # 2-d array, like: (days, 1), only close price as one attribute

    scaler = MinMaxScaler()
    # df[['Close]] instead of df['Close']
    scaled_close = scaler.fit_transform(close_prices)

    return scaled_close, scaler

def pack_sequences(data, seq_len):
    """
    Pack data into (samples_num - seq_len) sequences,
    each sequence has seq_len days(time steps), 
    and one label for the target next day to be predicted.

    :param data: 2-d array, like: (days, 1)
    :param seq_len: int, sequence length
    :return X: 3-d array, packed sentences
    :return y: 2-d array, corresponding labels
    """
    dataX, dataY = [], []
    for i in range(len(data) - seq_len):
        dataX.append(data[i : i + seq_len])
        dataY.append(data[i + seq_len])

    X = np.array(dataX)
    y = np.array(dataY)

    return X, y

def prepare_dataloader(X, y, batch_size=32, test_ratio=0.2):
    """
    Prepare DataLoader for training and testing.

    :param X: 3-d array, like: (samples_num, seq_len, 1)
    :param y: 2-d array, like: (samples_num, 1)
    :param batch_size: int
    :param split_ratio: float, ratio of testing data
    :return: trainLoader, testLoader, X_test, y_test
    """
    tensor_X = torch.from_numpy(X).float()
    tensor_y = torch.from_numpy(y).float()

    # use history data to train, then predict future data, so shuffle=False
    X_train, X_test, y_train, y_test = train_test_split(tensor_X, tensor_y, test_size=test_ratio, shuffle=False)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    

    return train_loader, test_loader, X_test, y_test

def evaluate(model, X_test, y_test, scaler):
    """
    Calculate MAE and RMSE for the model, so directly use X_test and y_test.

    :param model: LSTM model
    :param X_test: 3-d Tensor, like: (samples_num, seq_len, 1)
    :param y_test: 2-d Tensor, like: (samples_num, 1)
    :param scaler: MinMaxScaler, the one used before to scale data, to inverse transform
    :return: predictions, actuals, mae, rmse
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            # X_test: (test_samples_num, seq_len, 1)
            # X_test[i]: (seq_len, 1) -> to put into LSTM, should unsqueeze
            input = X_test[i].unsqueeze(0)  # Tensor: (batch_size=1, seq_len, 1)
            out = model(input)  # Tensor: (batch_size=1, 1)
            pred = out.item()
            actual = y_test[i].item()
            predictions.append(pred)
            actuals.append(actual)

    # inverse transform to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))  # (samples_num, 1)
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))  # (samples_num, 1)

    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)

    return predictions, actuals, mae, rmse

def plot_curve(predictions, actuals, save_path):
    """
    Plot the curve of predictions and actuals.

    :param predictions: 2-d np.array, (samples_num, 1)
    :param actuals: 2-d np.array, (samples_num, 1)
    """
    plt.figure(figsize=(10, 6))
    predictions = predictions.flatten()
    actuals = actuals.flatten()
    plt.plot(predictions, label='Prediction', color='red')
    plt.plot(actuals, label='Actual', color='blue')
    plt.title('LSTM Stock Price Prediction')
    plt.xlabel('Days (Time Steps)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    # Prepare data
    price_data, scaler= load_and_prepare_data('Microsoft_stock_history.csv')
    seq_len = 60
    X, y = pack_sequences(price_data, seq_len)
    train_loader, test_loader, X_test, y_test = prepare_dataloader(X, y, batch_size=32, test_ratio=0.2)
    
    # Initialize model
    model = LSTMModel(input_size=1, hidden_size=64, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(10):
        train(model, train_loader, epoch, criterion, optimizer)

    # Evaluate
    predictions, actuals, mae, rmse = evaluate(model, X_test, y_test, scaler)
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    plot_curve(predictions, actuals, 'LSTM_stock_prediction.png')

if __name__ == '__main__':
    main()

    

    


    




