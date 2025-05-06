
import torch
import pandas as pd
from models.bilstm_attention import AttentionBiLSTM
from torch.utils.data import DataLoader, TensorDataset

# Load processed test data
X_test = torch.load("X_test.pt")
y_test = torch.load("y_test.pt")

# Load trained model
model = AttentionBiLSTM(input_dim=X_test.shape[2], hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("bilstm_attention_model.pt"))
model.eval()

# Predict
with torch.no_grad():
    predictions = model(X_test).squeeze().numpy()
    actuals = y_test.numpy()

# Save to CSV
pd.DataFrame(actuals, columns=["Actual"]).to_csv("y_test.csv", index=False)
pd.DataFrame(predictions, columns=["Predicted"]).to_csv("y_pred.csv", index=False)

print("Saved y_test.csv and y_pred.csv")
