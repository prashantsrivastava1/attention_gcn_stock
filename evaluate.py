import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.lstm_attention import LSTMAttention
from train import X_temp, y, temporal_features
import joblib

# Load scaler and model
target_scaler = joblib.load("data/target_scaler.save")
model = LSTMAttention(input_dim=X_temp.shape[2], hidden_dim=64)
model.load_state_dict(torch.load("data/lstm_attention_model.pth"))
model.eval()

# Predict and collect attention
all_preds, all_trues, all_weights = [], [], []
with torch.no_grad():
    for i in range(len(X_temp)):
        x = X_temp[i:i+1]  # [1, seq_len, input_dim]
        y_true = y[i].unsqueeze(0)
        context, attn, _ = model(x)
        output = nn.Linear(context.shape[1], 1)(context)
        all_preds.append(output.item())
        all_trues.append(y_true.item())
        all_weights.append(attn.squeeze(0).numpy())

# Inverse transform predictions
y_pred_inv = target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))
y_true_inv = target_scaler.inverse_transform(np.array(all_trues).reshape(-1, 1))

# Metrics
mse = mean_squared_error(y_true_inv, y_pred_inv)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
r2 = r2_score(y_true_inv, y_pred_inv)
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_true_inv, label='True', color='blue')
plt.plot(y_pred_inv, label='Predicted', color='orange')
plt.title("VEDL Stock Price Prediction (Next Day)")
plt.xlabel("Time Index")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Average attention over dataset
avg_attention = np.mean(np.array(all_weights), axis=0)
plt.figure(figsize=(10, 4))
sns.heatmap(avg_attention.reshape(1, -1), cmap="YlGnBu", cbar=True,
            xticklabels=list(range(1, len(avg_attention)+1)), yticklabels=['Avg Attention'])
plt.title("Average Temporal Attention over Dataset")
plt.xlabel("Time Step")
plt.tight_layout()
plt.show()