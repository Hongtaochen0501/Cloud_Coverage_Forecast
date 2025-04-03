import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, target_idx=0, seq_len=3):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)

def run_lstm_model(df, seq_len=3, hidden_size=32, epochs=100):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["I_allsky", "I_clearsky", "Cloud_Amount"]])
    X_np, y_np = create_sequences(scaled, 0, seq_len)
    split = int(len(X_np) * 0.8)
    X_train = torch.tensor(X_np[:split], dtype=torch.float32)
    y_train = torch.tensor(y_np[:split], dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_np[split:], dtype=torch.float32)
    y_test = torch.tensor(y_np[split:], dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8)

    model = LSTMModel(input_size=3, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze().numpy()
        y_true = y_test.squeeze().numpy()

    y_pred_inv = scaler.inverse_transform(np.hstack([y_pred.reshape(-1,1), np.zeros((len(y_pred), 2))]))[:, 0]
    y_true_inv = scaler.inverse_transform(np.hstack([y_true.reshape(-1,1), np.zeros((len(y_true), 2))]))[:, 0]
    aligned_index = df.index[seq_len + split:]
    return y_pred_inv, y_true_inv, aligned_index
