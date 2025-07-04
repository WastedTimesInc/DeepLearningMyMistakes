import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# --- NEW: Create sequences of length seq_len ---
seq_len = 20

def create_sequences(X, y, seq_len):
    sequences = []
    labels = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i+seq_len])
        labels.append(y[i+seq_len-1])  # label aligned to last timestep in sequence
    return np.array(sequences), np.array(labels)

X_seq, y_seq = create_sequences(X, y, seq_len)

# --- 4. Manual train/val split (80% train, 20% val, no shuffle) ---
n = len(y_seq)
train_size = int(n * 0.8)
X_train, X_val = X_seq[:train_size], X_seq[train_size:]
y_train, y_val = y_seq[:train_size], y_seq[train_size:]

# --- 5. Dataset & DataLoader (updated for sequences) ---
class CryptoSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # shape: (samples, seq_len, features)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = CryptoSequenceDataset(X_train, y_train)
val_ds = CryptoSequenceDataset(X_val, y_val)
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=6, pin_memory=True , persistent_workers=True)
val_dl = DataLoader(val_ds, batch_size=512, num_workers=6, pin_memory=True, persistent_workers=True)

# --- 6. LSTM model with 2 layers and dropout ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        # take hidden state of last layer at last timestep
        last_hidden = hn[-1]  # (batch, hidden_dim)
        out = self.fc(last_hidden)
        return out

device = torch.device('cuda')
model = LSTMClassifier(input_dim=X.shape[1], hidden_dim=128,
                       num_classes=len(unique_labels), num_layers=2, dropout=0.3).to(device)

y_train_tensor = torch.from_numpy(y_train)
num_classes = len(unique_labels)
class_counts = torch.zeros(num_classes, dtype=torch.float32)
for i in range(num_classes):
    class_counts[i] = (y_train_tensor == i).sum()

class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.max()

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

epochs = 100
for epoch in range(epochs):
    train_loss = train_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_dl, criterion)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Acc={val_acc:.7f}")

torch.save(model.state_dict(), 'crypto_trading_model_lstm.pth')
