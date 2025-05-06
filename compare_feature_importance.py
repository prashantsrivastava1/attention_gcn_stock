import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from models.lstm_attention import LSTMAttention
from scipy.stats import spearmanr

# Load data
df = pd.read_csv("data/final_merged_data.csv")
temporal_features = [col for col in df.columns if col.startswith(('Open_', 'High_', 'Low_', 'Close_', 'Adj Close_', 'Volume_'))]
target_column = 'Close_VEDL'

# Normalize features and target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[temporal_features])
target_scaler = joblib.load("data/target_scaler.save")
y_scaled = target_scaler.transform(df[[target_column]])

# Prepare sequence data
seq_len = 20
X_seq = []
for i in range(seq_len, len(df) - 1):
    X_seq.append(X_scaled[i - seq_len:i])

X_seq = np.array(X_seq)  # [samples, seq_len, features]
X_tensor = torch.tensor(X_seq, dtype=torch.float32)

# === Linear Regression Coefficients ===
X_flat = X_scaled[seq_len:-1]
y_flat = y_scaled[seq_len + 1:]
lr = LinearRegression()
lr.fit(X_flat, y_flat)
linear_importance = np.abs(lr.coef_[0])
linear_norm = linear_importance / np.sum(linear_importance)

# === Attention Feature Attribution ===
model = LSTMAttention(input_dim=X_tensor.shape[2], hidden_dim=64)
checkpoint = torch.load("data/lstm_attention_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

attn_weights_total = np.zeros(len(temporal_features))

with torch.no_grad():
    for i in range(0, len(X_tensor), 32):
        batch = X_tensor[i:i+32]
        context, attn_weights, _ = model(batch)  # attn_weights: [B, T]
        weighted_inputs = attn_weights.unsqueeze(-1) * batch  # [B, T, F]
        feature_scores = weighted_inputs.sum(dim=1).sum(dim=0).numpy()  # [F]
        attn_weights_total += feature_scores

attn_norm = attn_weights_total / np.sum(attn_weights_total)

# === Comparison ===
df_compare = pd.DataFrame({
    "Feature": temporal_features,
    "Linear_Coef": linear_importance,
    "Linear_Coef_Norm": linear_norm,
    "Attention_Score": attn_weights_total,
    "Attention_Score_Norm": attn_norm
})

# Save to CSV
df_compare.to_csv("data/linear_vs_attention_importance.csv", index=False)

# Plot
df_plot = df_compare.sort_values("Attention_Score_Norm", ascending=False).head(10)
plt.figure(figsize=(10, 5))
plt.barh(df_plot["Feature"], df_plot["Attention_Score_Norm"], label="Attention-Based")
plt.barh(df_plot["Feature"], df_plot["Linear_Coef_Norm"], alpha=0.5, label="Linear Coef")
plt.xlabel("Normalized Feature Importance")
plt.title("Top 10 Feature Importances (Attention vs Linear)")
plt.legend()
plt.tight_layout()
plt.show()

# Rank correlation
rank_corr, _ = spearmanr(df_compare["Linear_Coef_Norm"], df_compare["Attention_Score_Norm"])
print(f"\n🔁 Spearman Rank Correlation: {rank_corr:.4f}")
