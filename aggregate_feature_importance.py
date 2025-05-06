import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_attention import LSTMAttention
from train import X_temp, temporal_features

# Load model and regressor
model = LSTMAttention(input_dim=X_temp.shape[2], hidden_dim=64)
checkpoint = torch.load("data/lstm_attention_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# DataLoader
dataset = TensorDataset(X_temp)
dataloader = DataLoader(dataset, batch_size=32)

# Aggregate attention-weighted feature contributions
feature_contributions = np.zeros(X_temp.shape[2])
total_weight = 0

with torch.no_grad():
    for (batch_x,) in dataloader:
        context, attn_weights, _ = model(batch_x)  # attn_weights: [batch, seq_len]
        batch_weights = attn_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        weighted_inputs = batch_weights * batch_x  # [batch, seq_len, input_dim]
        batch_feature_scores = weighted_inputs.sum(dim=1).sum(dim=0).numpy()  # [input_dim]
        feature_contributions += batch_feature_scores
        total_weight += batch_x.shape[0]

# Normalize
feature_importance = feature_contributions / total_weight
feature_importance /= feature_importance.sum()

# Plot
plt.figure(figsize=(12, 5))
sns.barplot(x=temporal_features, y=feature_importance)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Normalized Attribution Score")
plt.title("Aggregated Feature Importance from Attention")
plt.tight_layout()
plt.show()