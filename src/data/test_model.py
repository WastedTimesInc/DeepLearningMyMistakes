# Updated script with L2 regularization and LR scheduler

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

# --- 1. Load datasets ---
labels_df = pd.read_parquet('./processed/labels/BTCUSDT_2020-01-01_2025-07-01_L20_P0.0020.parquet')
features_df = pd.read_parquet('./processed/dataset/BTCUSDT_2020-01-01_2025-07-01_features.parquet')
labels_df.rename(columns={'opentime': 'OpenTime'}, inplace=True)
data_df = pd.merge(features_df, labels_df[['OpenTime', 'label']], on='OpenTime', how='inner')

# --- 2. Manual label encoding ---
unique_labels = sorted(data_df['label'].unique())
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
data_df['label_enc'] = data_df['label'].map(label_to_int).astype(np.int64)
print(f"Labels mapping: {label_to_int}")

# --- 3. Prepare features and labels arrays ---
drop_cols = ['OpenTime', 'label', 'openposition']
feature_cols = [c for c in data_df.columns if c not in drop_cols + ['label_enc']]
X = data_df[feature_cols].values.astype(np.float32)
y = data_df['label_enc'].values.astype(np.int64)

# --- 4. Create sequences ---
seq_len = 20

def create_sequences(X, y, seq_len):
    sequences, labels = [], []
    for i in range(len(X) - seq_len + 1):
        seq = X[i:i+seq_len]
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue
        sequences.append(seq)
        labels.append(y[i+seq_len-1])
    return np.array(sequences), np.array(labels)

X_seq, y_seq = create_sequences(X, y, seq_len)

# --- 5. Train/val split ---
n = len(y_seq)
train_size = int(n * 0.8)
X_train, X_val = X_seq[:train_size], X_seq[train_size:]
y_train, y_val = y_seq[:train_size], y_seq[train_size:]

# --- 6. Dataset & Dataloader ---
class CryptoSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_dl = DataLoader(CryptoSequenceDataset(X_train, y_train), batch_size=1024, shuffle=True, num_workers=6, pin_memory=True)
val_dl = DataLoader(CryptoSequenceDataset(X_val, y_val), batch_size=1024, num_workers=6, pin_memory=True)

# --- 7. LSTM Model ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(hn[-1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(input_dim=X.shape[1], hidden_dim=256, num_classes=len(unique_labels)).to(device)

# --- 8. Load existing model ---
model_dir = "./saved_models"
os.makedirs(model_dir, exist_ok=True)
existing_models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]

if existing_models:
    print("Available models:")
    for i, name in enumerate(existing_models):
        print(f"[{i}] {name}")
    choice = input("Enter the number of the model to load (or leave blank to start new): ")
    if choice.strip().isdigit() and int(choice) < len(existing_models):
        selected = existing_models[int(choice)]
        model.load_state_dict(torch.load(os.path.join(model_dir, selected)))
        print(f"Loaded model: {selected}")
    else:
        print("Training new model...")
else:
    print("No saved models found. Training new model...")

# --- 9. Loss, Optimizer, Scheduler ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)  # L2 Regularization
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# --- 10. Train/eval functions ---
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# --- 11. Training loop with interrupt saving ---
epochs = 100
try:
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dl, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_dl, criterion)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Acc={val_acc:.6f}")
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving model...")
    name = input("Enter name for this model version (without extension): ")
    torch.save(model.state_dict(), os.path.join(model_dir, f"{name}.pth"))
    print(f"Model saved as {name}.pth")
    exit()

# --- Final save ---
final_name = input("Training completed. Save final model as (without extension): ")
torch.save(model.state_dict(), os.path.join(model_dir, f"{final_name}.pth"))
print(f"Final model saved as {final_name}.pth")
