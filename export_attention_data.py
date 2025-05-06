import torch
import numpy as np
import pandas as pd
from models.lstm_attention import LSTMAttention
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("data/final_merged_data.csv")
temporal_features = [col for col in df.columns if col.startswith(('Open_', 'High_', 'Low_', 'Close_', 'Adj Close_', 'Volume_'))]
target_column = 'Close_VEDL'

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[temporal_features])
target_scaler = joblib.load("data/target_scaler.save")

# Prepare sequences
seq_len = 20
X_seq = []
for i in range(seq_len, len(df) - 1):
    X_seq.append(X_scaled[i - seq_len:i])
X_seq = np.array(X_seq)
X_tensor = torch.tensor(X_seq, dtype=torch.float32)

# Load model
model = LSTMAttention(input_dim=X_tensor.shape[2], hidden_dim=64)
checkpoint = torch.load("data/lstm_attention_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Get attention scores
attention_all = []
with torch.no_grad():
    for i in range(0, len(X_tensor), 32):
        batch = X_tensor[i:i+32]
        _, attn_weights, _ = model(batch)
        attention_all.append(attn_weights.numpy())
attention_all = np.vstack(attention_all)

# Save both files
np.savetxt("data/X_seq.csv", X_seq.reshape(X_seq.shape[0], -1), delimiter=",")
np.savetxt("data/attention_scores.csv", attention_all, delimiter=",")
