#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyedflib
import numpy as np

def read_edf_signals(edf_file):
    f = pyedflib.EdfReader(edf_file)
    num_signals = f.signals_in_file
    signals = []
    for i in range(num_signals):
        signal = f.readSignal(i)
        signals.append(signal)

    labels = f.getSignalLabels()
    sampling_rates = f.getSampleFrequencies()

    f.close()
    
    return signals, labels, sampling_rates

edf_file_path = "./dataset/files/ucddb002.rec"
signals, labels, sampling_rates = read_edf_signals(edf_file_path)

print("Signal labels:", labels)
print("Sampling rates:", sampling_rates)

print("First 10 samples of the first signal:", signals[0][:10]) 

# Imp signals -> 'Flow', 'ribcage'


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(10, 6))
plt.plot(signals[8])
plt.plot(signals[10], 'r')
plt.xlabel('Time')
plt.ylabel('Flow/Ribcage')
plt.title('Flow and Ribcage signal')
plt.legend(['Flow', 'Ribcage'])


# In[3]:


import numpy as np
from sklearn.preprocessing import RobustScaler


print("Shape of signals:", signals[8].shape)
signals_stacked = np.column_stack([signals[8], signals[10]])

scaler = RobustScaler()
scaled_signals = scaler.fit_transform(signals_stacked)

flow_scaled = scaled_signals[:, 0]
ribcage_scaled = scaled_signals[:, 1]

# normalizer = Normalizer()
# X2 = normalizer.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.plot(flow_scaled)
plt.plot(ribcage_scaled, 'r', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Flow/Ribcage')
plt.title('Flow and Ribcage signal')
plt.legend(['Flow', 'Ribcage'])


plt.figure(figsize=(10, 6))
plt.plot(signals[8]/np.max(signals[8]))
plt.plot(signals[10]/np.max(signals[10]), 'r', alpha=0.5)   
plt.xlabel('Time')
plt.ylabel('Flow/Ribcage')
plt.title('Flow and Ribcage signal')
plt.legend(['Flow', 'Ribcage'])


# In[4]:


print("-------------------------------------------------------------------------------------------")
print("Distribution of sleep stages in all the files:")
for i in range(2, 10):
    try:
        sleep_stages_ = np.loadtxt(f"dataset/files/ucddb00{i}_stage.txt", dtype=int)
    except:
        continue
    print(i, np.unique(sleep_stages_, return_counts=True))


for i in range(1, 29):
    try:
        sleep_stages_ = np.loadtxt(f"dataset/files/ucddb0{i}_stage.txt", dtype=int)
    except:
        continue
    print(i, np.unique(sleep_stages_, return_counts=True))
print("-------------------------------------------------------------------------------------------")


sleep_stages_label = np.loadtxt(f"idk/files/ucddb005_stage.txt", dtype=int)
sleep_stages_label
times = pd.Index(np.arange(0, len(sleep_stages_label) * 30, 30))

fig, ax = plt.subplots(figsize=(10, 3))

ax.plot(times, sleep_stages_label, color='blue', label='Wake/Awake', linewidth=1)
ax.set_xlabel('Time (HH:MM:SS)', fontsize=12)
ax.set_ylabel('Sleep stages', fontsize=12)
ax.set_title('Sleep Stages for Subject 5', fontsize=14, pad=20)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#---------------------------------------------------
# Note: 0 denotes awake
fig, ax = plt.subplots(figsize=(10, 3))
wake_awake_label_signal = (sleep_stages_label > 0).astype(int)
np.unique(wake_awake_label_signal, return_counts=True)

ax.plot(times, wake_awake_label_signal, color='blue', label='Wake/Awake', linewidth=1)

ax.set_xlabel('Time (HH:MM:SS)', fontsize=12)
ax.set_ylabel('Sleep stages', fontsize=12)
ax.set_title('Wake/Awake Signal for Subject 5', fontsize=14, pad=20)

ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[5]:


x_list = []
y_list = []

for i in range(2, 29):
    try:
        if i<10:
            sleep_stages = np.loadtxt(f"dataset/files/ucddb00{i}_stage.txt", dtype=int)
            edf_file_path = f"./dataset/files/ucddb00{i}.rec"
            signals, labels, sampling_rates = read_edf_signals(edf_file_path)
        else:
            sleep_stages = np.loadtxt(f"dataset/files/ucddb0{i}_stage.txt", dtype=int)
            edf_file_path = f"./dataset/files/ucddb0{i}.rec"
            signals, labels, sampling_rates = read_edf_signals(edf_file_path)

        signals_stacked = np.column_stack([signals[8], signals[10]])
        scaler = RobustScaler()
        scaled_signals = scaler.fit_transform(signals_stacked)
        flow_scaled = scaled_signals[:, 0]
        ribcage_scaled = scaled_signals[:, 1]

        flow_scaled = flow_scaled[:len(sleep_stages)*240]
        ribcage_scaled = ribcage_scaled[:len(sleep_stages)*240]

        sleep_stages_remove_indices = []
        signal_remove_indices = []
        for j in range(len(sleep_stages)):
            if sleep_stages[j] == 8:
                sleep_stages_remove_indices.append(j)
                _indices = np.arange(j*240, (j+1)*240)
                signal_remove_indices.extend(_indices)
        
        flow_scaled = np.delete(flow_scaled, signal_remove_indices)
        ribcage_scaled = np.delete(ribcage_scaled, signal_remove_indices)
        sleep_stages = np.delete(sleep_stages, sleep_stages_remove_indices)

        reshaped1 = flow_scaled.reshape(-1, 240)
        reshaped2 = ribcage_scaled.reshape(-1, 240)

        sleep_stages = sleep_stages.reshape(-1, 1)
        flow_ribcage = np.stack([reshaped1, reshaped2], axis=1)
        x_list.append(flow_ribcage)
        y_list.append(sleep_stages)
    except:
        print(f"Error in file {i}")
        continue

print("-------------------------------------------------------------------------------------------")
for i in range(len(x_list)):
     print(f"File {i} has {len(x_list)} signals", \
        "\nsamples [nasal, ribcage]", f"[{len(signals[8])}, {len(signals[10])}]", \
        "\nsampling rate [nasal, ribcage]", f"[{sampling_rates[8]}, {sampling_rates[10]}]",\
        "\nno of sleep stages", len(x_list[i])* len(x_list[0][1][0])/(30*sampling_rates[8]), len(x_list[i])* len(x_list[0][1][0])/(30*sampling_rates[10]),  \
        "\nSleep stages: ", len(y_list[i])) 
     print()



# In[6]:


print(y_list[0].shape, x_list[0].shape)


# In[7]:


y_binary_list = []
for i in range(len(y_list)):
    y_binary = (y_list[i] > 0).astype(int)
    y_binary_list.append(y_binary)


# In[8]:


y_3_stage_list = []
for i in range(len(y_list)):
    # 0 -> wake, 1 -> REM, 2,3,4,5 -> NREM
    y_3_stage = np.zeros_like(y_list[i])
    y_3_stage[y_list[i] == 1] = 1 # REM
    y_3_stage[y_list[i] == 2] = 2
    y_3_stage[y_list[i] == 3] = 2
    y_3_stage[y_list[i] == 4] = 2
    y_3_stage[y_list[i] == 5] = 2

    # 0 -> wake, 1 -> REM, 2 -> NREM
    y_3_stage_list.append(y_3_stage)


# In[9]:


y_4_stage_list = []
for i in range(len(y_list)):
    # 0 -> wake, 1 -> REM, 2,3 -> Light Sleep, 4,5 -> Deep Sleep
    y_4_stage = np.zeros_like(y_list[i])
    y_4_stage[y_list[i] == 1] = 1 # REM
    y_4_stage[y_list[i] == 2] = 2
    y_4_stage[y_list[i] == 3] = 2
    y_4_stage[y_list[i] == 4] = 3
    y_4_stage[y_list[i] == 5] = 3

    # 0 -> wake, 1 -> REM, 2 -> Light Sleep, 3 -> Deep Sleep

    y_4_stage_list.append(y_4_stage)


# In[10]:


print(np.unique(y_3_stage_list[0], return_counts=True))
print(np.unique(y_binary_list[0], return_counts=True))
print(np.unique(y_4_stage_list[0], return_counts=True))
np.unique(y_list[0], return_counts=True)


# In[11]:


binary_subject_wise_data = list(zip(x_list, y_binary_list))
for i, data in enumerate(binary_subject_wise_data):
    train_data = [x for j, x in enumerate(binary_subject_wise_data) if j != i]
    test_data = data

    train_features = np.vstack([x[0] for x in train_data])
    train_labels = np.vstack([x[1] for x in train_data])
    test_features = test_data[0]
    test_labels = test_data[1]


# In[12]:


len(binary_subject_wise_data)


# In[13]:


X_train = np.vstack(x_list[:-1])
Y_train = np.vstack(y_binary_list[:-1])


X_test = x_list[-1]
Y_test = y_list[-1]
# Xdata.shape, Ydata.shape


# In[14]:


np.unique(Y_train, return_counts=True)


# In[ ]:


import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# -------------------------
# Positional Encoding Modules
# -------------------------

class tAPE(nn.Module):
    """
    Time Absolute Positional Encoding (tAPE)
    Equation (roughly based on ConvTran paper modifications).
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding table
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # incorporate d_model/max_len scaling to preserve distance awareness
        pe[:, 0::2] = torch.sin((position * div_term) * (d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class eRPE(nn.Module):
    """
    Efficient Relative Positional Encoding (eRPE)
    A simplified implementation inspired by the ConvTran paper.
    """
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super(eRPE, self).__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        # Create a table for relative biases (for distances from -(L-1) to L-1)
        self.relative_bias_table = nn.Parameter(torch.zeros(2 * seq_len - 1, num_heads))
        # Prepare relative index mapping (shape: [seq_len, seq_len])
        coords = torch.arange(seq_len)
        relative_coords = coords[None, :] - coords[:, None]  # shape: (seq_len, seq_len)
        relative_coords += seq_len - 1  # shift to non-negative
        self.register_buffer("relative_index", relative_coords)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)
        # Initialize the relative bias table
        nn.init.zeros_(self.relative_bias_table)

    def forward(self, x):
        # x: (batch, seq_len, emb_size)
        B, L, C = x.shape
        # Linear projections
        q = self.query(x).view(B, L, self.num_heads, -1).permute(0,2,1,3)  # (B, num_heads, L, d_head)
        k = self.key(x).view(B, L, self.num_heads, -1).permute(0,2,3,1)     # (B, num_heads, d_head, L)
        v = self.value(x).view(B, L, self.num_heads, -1).permute(0,2,1,3)   # (B, num_heads, L, d_head)
        # Scaled dot-product attention
        attn = torch.matmul(q, k) * self.scale  # (B, num_heads, L, L)
        attn = F.softmax(attn, dim=-1)
        # Gather relative biases using the pre-computed relative indices.
        # relative_index: (L, L) -> bias shape becomes (L, L, num_heads)
        relative_bias = self.relative_bias_table[self.relative_index.view(-1)].view(L, L, self.num_heads)
        # Permute to (1, num_heads, L, L) and add to attention scores
        relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_bias
        # Apply attention dropout
        attn = self.dropout(attn)
        # Compute output
        out = torch.matmul(attn, v)  # (B, num_heads, L, d_head)
        out = out.permute(0,2,1,3).contiguous().view(B, L, C)
        out = self.to_out(out)
        return out

# -------------------------
# Transformer Classifier Model
# -------------------------

class TransformerClassifier(nn.Module):
    def __init__(self, input_channels, seq_len, embed_dim=32, num_heads=4, num_layers=2,
                 num_classes=2, dropout=0.1):
        """
        input_channels: number of channels in the input (e.g. 2)
        seq_len: length of the time series (e.g. 240)
        embed_dim: dimension of the embedding space
        num_heads: number of attention heads
        num_layers: number of transformer encoder layers
        num_classes: number of output classes
        """
        super(TransformerClassifier, self).__init__()
        self.seq_len = seq_len
        # Project input (channels) to embed_dim for each time step
        self.input_proj = nn.Linear(input_channels, embed_dim)
        # Time Absolute Positional Encoding
        self.tape = tAPE(embed_dim, dropout=dropout, max_len=seq_len)
        # Transformer Encoder (PyTorch built-in)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Efficient Relative Positional Encoding
        self.erpe = eRPE(emb_size=embed_dim, num_heads=num_heads, seq_len=seq_len, dropout=dropout)
        # Pool over time dimension and classify
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        # Transpose to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Project to embedding dimension
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)
        # Add absolute positional encoding
        x = self.tape(x)
        # Transformer encoder expects (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # back to (batch, seq_len, embed_dim)
        # Apply relative positional encoding
        x = self.erpe(x)
        # Pool along the time dimension
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, embed_dim)
        logits = self.classifier(x)  # (batch, num_classes)
        return logits

# -------------------------
# Custom Dataset
# -------------------------

class SleepStageDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: numpy array of shape (num_samples, channels, seq_len)
        Y: numpy array of shape (num_samples, 1) or (num_samples,)
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long().squeeze()  # convert to 1D tensor if needed

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -------------------------
# Training and Evaluation Loops
# -------------------------

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        print(X_batch.shape, Y_batch.shape)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == Y_batch).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return epoch_loss, accuracy


# In[17]:


X_train.shape, Y_train.shape


# 

# In[18]:


num_samples = 20053
channels = 2
seq_len = 240
num_classes = 2

# Create datasets and dataloaders
train_dataset = SleepStageDataset(X_train, Y_train)
test_dataset = SleepStageDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Set up device, model, criterion, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = TransformerClassifier(input_channels=channels, seq_len=seq_len,
                                embed_dim=32, num_heads=4, num_layers=2,
                                num_classes=num_classes, dropout=0.1)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
for epoch in range(1, num_epochs+1):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch}/{num_epochs}: Train Loss = {train_loss:.4f} | Test Loss = {test_loss:.4f} | Test Accuracy = {test_acc:.4f}")


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb')

