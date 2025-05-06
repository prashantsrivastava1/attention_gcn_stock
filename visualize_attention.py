import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from models.lstm_attention import LSTMAttention
from train import X_temp, temporal_features

# Load sample batch
sample_input = X_temp[:1]  # shape [1, seq_len, input_dim]

# Load model
model = LSTMAttention(input_dim=sample_input.shape[2], hidden_dim=64)
model.eval()

# Forward pass
with torch.no_grad():
    _, attn_weights, lstm_out = model(sample_input)

# Convert to numpy
attn_weights = attn_weights.squeeze(0).numpy()  # [seq_len]
lstm_out = lstm_out.squeeze(0).numpy()          # [seq_len, hidden_dim*2]

# Compute weighted output per time step
weighted_output = (attn_weights[:, np.newaxis] * lstm_out)  # [seq_len, hidden_dim*2]
avg_contribution = weighted_output.mean(axis=0)  # [hidden_dim*2]

# To relate features, we compute input importance by aggregating across time
feature_importance = sample_input.squeeze(0).numpy().T @ attn_weights  # [input_dim]

# Normalize and plot
feature_importance /= feature_importance.sum()
plt.figure(figsize=(10, 4))
sns.barplot(x=temporal_features, y=feature_importance)
plt.xticks(rotation=45, ha='right')
plt.title("Feature-wise Attention Contribution")
plt.tight_layout()
plt.show()

# Temporal heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(attn_weights.reshape(1, -1), cmap="YlGnBu", cbar=True,
            xticklabels=list(range(1, len(attn_weights)+1)), yticklabels=['Attention'])
plt.title("Temporal Attention Weights across Time Steps")
plt.xlabel("Time Step")
plt.ylabel("")
plt.tight_layout()
plt.show()
