import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_attention import LSTMAttention
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load merged data
df = pd.read_csv("data/final_merged_data.csv")

# Feature selection
temporal_features = [col for col in df.columns if col.startswith(('Open_', 'High_', 'Low_', 'Close_', 'Adj Close_', 'Volume_'))]
target_column = 'Close_VEDL'

# Normalize and prepare data
scaler = StandardScaler()
temporal_data = scaler.fit_transform(df[temporal_features])

target_scaler = StandardScaler()
scaled_target = target_scaler.fit_transform(df[[target_column]])

# Sequence preparation
temporal_seq_len = 20
X_temp = []
y = []
for i in range(temporal_seq_len, len(df) - 1):  # Predict next day's price
    X_temp.append(temporal_data[i - temporal_seq_len:i])
    y.append(scaled_target[i + 1][0])

X_temp = torch.tensor(np.array(X_temp), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

# Dataloader
dataset = TensorDataset(X_temp, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = LSTMAttention(input_dim=X_temp.shape[2], hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Output head
regressor = nn.Linear(64 * 2, 1)  # Because it's bidirectional LSTM

# Training loop
epochs = 100
model.train()
regressor.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        context, _, _ = model(batch_x)
        output = regressor(context)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Save scaler and model
joblib.dump(target_scaler, "data/target_scaler.save")
torch.save({
    "model_state_dict": model.state_dict(),
    "regressor_state_dict": regressor.state_dict()
}, "data/lstm_attention_model.pth")
