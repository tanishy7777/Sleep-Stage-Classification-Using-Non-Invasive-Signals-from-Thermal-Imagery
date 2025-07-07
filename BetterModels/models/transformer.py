#!/usr/bin/env python
# coding: utf-8

# In[1]: Imports
import pyedflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math # Needed for Transformer Positional Encoding

from sklearn.preprocessing import RobustScaler
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split # Keep for potential future use
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
# from imblearn.over_sampling import SMOTE # SMOTE not used, but kept if needed later
import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for Transformer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.special import softmax # Used in plotter, keep


# --- Configuration ---
DATASET_DIR = "./dataset/files/" # ADJUST THIS PATH if needed
OUTPUT_DIR = "./processed_data_transformer" # Use a different output dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
EXPECTED_SR = 8 # Define expected sampling rate globally
SAMPLES_PER_EPOCH = int(30 * EXPECTED_SR) # 240

# --- EDF Reading Function (from original code) ---
def read_edf_signals(edf_file):
    # ... (Keep the exact same function as in the previous CNN-LSTM version) ...
    """Reads signals, labels, and sampling rates from an EDF file."""
    try:
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
    except Exception as e:
        print(f"Error reading {edf_file}: {e}")
        return None, None, None

# In[2]: Transformer Model Definition

# --- Positional Encoding Modules (from original Transformer code) ---
class tAPE(nn.Module):
    """Time Absolute Positional Encoding (tAPE)"""
    def __init__(self, d_model, dropout=0.1, max_len=512): # Increased max_len default
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
             raise ValueError(f"Input sequence length ({seq_len}) exceeds tAPE max_len ({self.pe.size(1)})")
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# Note: eRPE implementation is omitted here for simplicity, relying on PyTorch's built-in TransformerEncoder.
# You could add the eRPE class back and integrate it if desired.

# --- Transformer Classifier Model ---
class TransformerClassifier(nn.Module):
    def __init__(self, input_channels, seq_len, embed_dim=64, num_heads=4, num_layers=2,
                 num_classes=2, dropout=0.1, d_ff=128): # Added d_ff (feed-forward dim)
        """
        Transformer Encoder based classifier.
        Args:
            input_channels (int): Number of input features (e.g., 2 for Flow, Ribcage).
            seq_len (int): Length of the input sequence (e.g., 240).
            embed_dim (int): Dimension of the embedding space (d_model).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
            d_ff (int): Dimension of the feed-forward layer in the encoder.
        """
        super(TransformerClassifier, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Input projection: From input_channels per time step to embed_dim
        # Using Conv1d is common for time series to capture local patterns before embedding
        self.input_proj = nn.Conv1d(input_channels, embed_dim, kernel_size=1)
        # Alternatively, a Linear layer after reshaping:
        # self.input_proj = nn.Linear(input_channels, embed_dim)

        # Time Absolute Positional Encoding
        self.tape = tAPE(embed_dim, dropout=dropout, max_len=seq_len + 1) # +1 just in case

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True # IMPORTANT: Expects (batch, seq, feature)
        )
        # Transformer Encoder Stack
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier Head
        # Option 1: Use the output of the [CLS] token (if added)
        # Option 2: Average pool the output sequence
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Optional: Layer Normalization before classifier
        self.norm_out = nn.LayerNorm(embed_dim)


    def forward(self, x):
        # Input x shape: (batch, channels, seq_len), e.g., (B, 2, 240)

        # --- Input Projection ---
        # Conv1d expects (batch, channels, seq_len)
        x = self.input_proj(x) # Shape: (batch, embed_dim, seq_len)
        x = x.permute(0, 2, 1) # Shape: (batch, seq_len, embed_dim) - Required by batch_first=True encoder

        # --- Add Positional Encoding ---
        x = self.tape(x) # Shape: (batch, seq_len, embed_dim)

        # --- Transformer Encoder ---
        # Input shape expected: (batch, seq_len, embed_dim) due to batch_first=True
        x = self.transformer_encoder(x) # Output shape: (batch, seq_len, embed_dim)

        # --- Classification Head ---
        # Average Pooling Option:
        # Permute for pooling: (batch, embed_dim, seq_len)
        x_pooled = x.permute(0, 2, 1)
        x_pooled = self.pool(x_pooled).squeeze(-1) # Shape: (batch, embed_dim)

        # Optional Layer Norm
        x_norm = self.norm_out(x_pooled)

        logits = self.classifier(x_norm) # Shape: (batch, num_classes)

        return logits


# In[3]: Helper Functions (Preprocessing, Training, Testing, Metrics, Plotting)

# --- Preprocessing Helper Functions ---
def normalize_channels(data):
    # ... (Keep the exact same function as in the previous CNN-LSTM version) ...
    """Applies Z-score normalization independently to each channel."""
    norm_data = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[1]): # Iterate through channels
        channel_data = data[:, i, :]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 1e-6: # Avoid division by zero
            norm_data[:, i, :] = (channel_data - mean) / std
        else:
            norm_data[:, i, :] = channel_data # Keep as is if std is zero
    return norm_data


def apply_sg_filter(data, window=25, order=5):
     # ... (Keep the exact same function as in the previous CNN-LSTM version) ...
    """Applies Savitzky-Golay filter to each channel of each sample."""
    seq_len = data.shape[2]
    if window > seq_len: window = seq_len - 1 if seq_len % 2 == 0 else seq_len
    if window % 2 == 0: window -= 1
    if window < order + 1: return data
    if window <= 0: return data

    filtered = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]): # Samples
        for j in range(data.shape[1]): # Channels
            try:
                 filtered[i, j, :] = savgol_filter(data[i, j, :], window, order)
            except ValueError: # Handle potential errors
                 filtered[i, j, :] = data[i, j, :]
    return filtered


# --- Training and Testing Loops ---
# These loops are generic and should work with the Transformer model
def train_loop(model, num_epochs, criterion, optimizer, train_loader, device):
    # ... (Keep the exact same function as in the previous CNN-LSTM version) ...
    """Trains the model for a specified number of epochs."""
    train_results_logits = [] # Store raw logits for potential later analysis
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) # Move data to device
            labels = labels.long() # Ensure labels are Long type for CrossEntropyLoss

            optimizer.zero_grad()   # Zero gradients
            outputs = model(inputs) # Forward pass (get logits)
            loss = criterion(outputs, labels) # Calculate loss

            # Store labels and logits for this batch
            train_results_logits.append((labels.detach().cpu(), outputs.detach().cpu()))

            loss.backward()         # Backward pass
            optimizer.step()        # Update weights

            total_loss += loss.item()

            # Print progress within the epoch
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}", end='\r')

        avg_epoch_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs} completed. Average Training Loss: {avg_epoch_loss:.4f}") # Newline after epoch completes

    print("Training finished.")
    return train_results_logits # Return list of (labels_batch, logits_batch) tuples


def test_loop(model, criterion, test_loader, device):
    # ... (Keep the exact same function as in the previous CNN-LSTM version) ...
    """Evaluates the model on the test set."""
    test_results = [] # Store (true_labels, predicted_logits) tuples
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculations
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            outputs = model(inputs) # Forward pass (get logits)
            loss = criterion(outputs, labels) # Calculate loss
            total_loss += loss.item()

            # Store true labels and predicted logits
            test_results.append((labels.cpu(), outputs.cpu()))

            # Calculate accuracy for progress (optional but helpful)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Test Evaluation Complete. Average Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return test_results # Return list of (labels_batch, logits_batch) tuples


# --- Metrics Calculation ---
def calculate_multiclass_specificity(cm):
     # ... (Keep the exact same function as in the previous CNN-LSTM version) ...
    """Calculates per-class and weighted specificity from a confusion matrix."""
    n_classes = cm.shape[0]
    per_class_specificity = []
    supports = []
    total_samples = cm.sum()

    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total_samples - (tp + fp + fn)

        denominator = (tn + fp)
        spec_k = tn / denominator if denominator > 0 else 0.0
        per_class_specificity.append(spec_k)
        support_k = cm[k, :].sum()
        supports.append(support_k)

    weighted_specificity = np.sum(np.array(per_class_specificity) * np.array(supports)) / total_samples if total_samples > 0 else 0.0
    return per_class_specificity, weighted_specificity, supports

def load_metrics(test_results, fold_index, class_labels=None, plot_indices=[0, 5, 10, 15, 20, 24]):
    # ... (Keep the exact same function as in the previous CNN-LSTM version, it's model agnostic) ...
    """Calculates and prints metrics from test results (list of label/logit tuples)."""
    if not test_results:
        print("Warning: No test results to calculate metrics from.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])

    y_true_batches = [batch[0] for batch in test_results]
    y_logits_batches = [batch[1] for batch in test_results]

    y_true = torch.cat(y_true_batches).numpy()
    y_logits = torch.cat(y_logits_batches)
    y_prob = torch.softmax(y_logits, dim=1).numpy()
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if class_labels and len(class_labels) != n_classes:
        print(f"Warning: Mismatch class_labels ({len(class_labels)}) vs predicted classes ({n_classes}).")
        class_labels = None

    print(f"\n--- Fold {fold_index+1} Metrics ---")
    print("Confusion Matrix:\n", cm)

    display_labels_cm = class_labels if class_labels else None
    if fold_index in plot_indices:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels_cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Fold {fold_index+1} Confusion Matrix')
        plt.show()

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) # Macro Recall/Sensitivity
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_specificity, weighted_specificity, _ = calculate_multiclass_specificity(cm)
    specificity = weighted_specificity # Use weighted average for reporting

    auc_macro = np.nan
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Overwrite with binary specificity
        y_prob_positive = y_prob[:, 1]
        try: auc_macro = roc_auc_score(y_true, y_prob_positive)
        except ValueError as e: print(f"Binary AUC Error: {e}")
    else: # Multi-class
        try: auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError as e: print(f"Multi-class AUC Error: {e}")

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    if n_classes == 2:
        print(f"Precision: {precision_macro:.4f}") # Macro is same as binary here
        print(f"Recall (Sensitivity): {recall_macro:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1-Score: {f1_macro:.4f}")
        print(f"AUC-ROC: {auc_macro:.4f}")
    else:
        target_names = class_labels if class_labels else [f"Class {i}" for i in range(n_classes)]
        print("\nMulti-class classification report:")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        print("\n--- Specificity Calculation ---")
        for idx, spec in enumerate(per_class_specificity): print(f"Specificity ({target_names[idx]}): {spec:.4f}")
        print(f"Weighted Average Specificity: {weighted_specificity:.4f}")
        print("-----------------------------")
        print(f"\nAUC-ROC (Macro OvR): {auc_macro:.4f}")
        print(f"\n--- Macro Averaged Metrics ---")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall/Sensitivity (Macro): {recall_macro:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print("------------------------")

    print(f"--- End Fold {fold_index+1} Metrics ---")
    return accuracy, precision_macro, recall_macro, specificity, f1_macro, auc_macro, y_true, y_pred

# --- Plotting Helper ---
def plotter(test_results, fold_index, num_batches_to_plot=4, plot_indices=[0, 5, 10, 15, 20, 24]):
    # ... (Keep the exact same function as in the previous CNN-LSTM version) ...
    """Plots true vs predicted labels for a few batches from a specific fold."""
    if fold_index not in plot_indices or not test_results: return

    y_true_batches = [batch[0].cpu().numpy() for batch in test_results]
    y_logits_batches = [batch[1].cpu().numpy() for batch in test_results]
    y_pred_batches = [np.argmax(softmax(logits, axis=1), axis=1) for logits in y_logits_batches]

    num_available_batches = len(y_true_batches)
    batches_to_plot = min(num_batches_to_plot, num_available_batches)
    if batches_to_plot == 0: return

    nrows = int(np.ceil(np.sqrt(batches_to_plot)))
    ncols = int(np.ceil(batches_to_plot / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for i in range(batches_to_plot):
        batch_idx = min(i, num_available_batches - 1)
        y_true = y_true_batches[batch_idx]
        y_pred = y_pred_batches[batch_idx]
        x_axis = np.arange(len(y_true))

        ax = axes[i]
        ax.plot(x_axis, y_true, color='steelblue', linewidth=1.5, label='True', marker='o', linestyle='-', markersize=4)
        ax.plot(x_axis, y_pred, color='red', linewidth=1.5, label='Pred', alpha=0.7, marker='x', linestyle='--', markersize=4)
        ax.set_xlabel(f'Sample Index in Batch {batch_idx+1}', fontsize=10)
        ax.set_ylabel('Sleep Stage Label', fontsize=10)
        ax.set_title(f'Fold {fold_index+1} - Batch {batch_idx+1}', fontsize=12)
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        ax.set_yticks(unique_labels)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    for j in range(batches_to_plot, len(axes)): fig.delaxes(axes[j])
    fig.suptitle(f'True vs. Predicted Sleep Stages for Fold {fold_index+1}', fontsize=16, y=1.02)
    fig.tight_layout()
    plt.show()


# In[4]: Main Execution Function (`run`) using Transformer

# --- Hyperparameters for Transformer ---
INPUT_FEATURES = 5      # Flow and Ribcage
SEQ_LEN = SAMPLES_PER_EPOCH # 240
EMBED_DIM = 64          # d_model
NUM_HEADS = 4           # Attention heads
NUM_LAYERS = 3          # Transformer Encoder layers
D_FF = 128              # Dimension of feed-forward layer
DROPOUT = 0.2           # Dropout rate

# --- Training Hyperparameters ---
BATCH_SIZE = 64         # Increased batch size for potentially faster training
NUM_EPOCHS = 20         # Number of training epochs per fold
LEARNING_RATE = 0.0005  # Adam learning rate

def run(npz_file_path, class_labels):
    """Loads data, runs LOSO-CV with Transformer, trains, evaluates, reports metrics."""
    print(f"\n{'='*20} Starting Transformer Run for: {npz_file_path} {'='*20}")
    print(f"Class Labels: {class_labels}")
    n_classes = len(class_labels)

    # Load the dataset
    try:
        loaded_data = np.load(npz_file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: NPZ file not found at {npz_file_path}")
        print(f"Please ensure the file was created in the '{OUTPUT_DIR}' directory (or run the preprocessing block).")
        return

    num_folds = 0
    while f'X_{num_folds}' in loaded_data:
        num_folds += 1

    if num_folds == 0:
        print("Error: No data found in the NPZ file.")
        return

    print(f"Found data for {num_folds} subjects/folds.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_fold_metrics = []
    all_fold_predictions = [] # Store y_true and y_pred for combined CM

    # Leave-One-Subject-Out Cross-validation loop
    for i in range(num_folds):
        print(f"\n===== FOLD {i+1}/{num_folds} (Testing on Subject {i}) =====")

        # Prepare Train/Test data for this fold
        train_features_list = [loaded_data[f'X_{j}'] for j in range(num_folds) if j != i]
        train_labels_list = [loaded_data[f'Y_{j}'] for j in range(num_folds) if j != i]
        test_features = loaded_data[f'X_{i}']
        test_labels = loaded_data[f'Y_{i}']

        # Concatenate training data if more than one training subject exists
        if train_features_list:
            train_features = np.vstack(train_features_list)
            train_labels = np.vstack(train_labels_list)
        else: # Handle case with only 1 subject (train=0, test=1)
            print("Warning: Only one subject found. Training cannot proceed.")
            continue # Or handle as appropriate

        print(f"Train shapes: X={train_features.shape}, Y={train_labels.shape}")
        print(f"Test shapes: X={test_features.shape}, Y={test_labels.shape}")

        # Preprocess the data (Normalization + Filtering)
        # Apply per fold to avoid data leakage from test set stats into training
        train_norm = normalize_channels(train_features)
        train_sg = apply_sg_filter(train_norm)
        test_norm = normalize_channels(test_features)
        test_sg = apply_sg_filter(test_norm)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(train_sg, dtype=torch.float32)
        y_train_tensor = torch.tensor(train_labels, dtype=torch.long).squeeze() # Use squeeze() for CrossEntropyLoss
        X_test_tensor = torch.tensor(test_sg, dtype=torch.float32)
        y_test_tensor = torch.tensor(test_labels, dtype=torch.long).squeeze()

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # drop_last can help stabilize training
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # --- Initialize the Transformer model for this fold ---
        model = TransformerClassifier(
            input_channels=INPUT_FEATURES,
            seq_len=SEQ_LEN,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            num_classes=n_classes,
            dropout=DROPOUT,
            d_ff=D_FF
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- Train the model ---
        print("--- Training ---")
        train_loop(model, NUM_EPOCHS, criterion, optimizer, train_loader, device)

        # --- Evaluate the model ---
        print("\n--- Evaluating ---")
        test_results = test_loop(model, criterion, test_loader, device)

        # --- Calculate and store metrics for this fold ---
        accuracy, precision, recall, specificity, f1, auc, y_true_fold, y_pred_fold = load_metrics(
            test_results, i, class_labels=class_labels
        )

        fold_metrics = {
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1, "roc_auc": auc
        }
        all_fold_metrics.append(fold_metrics)
        all_fold_predictions.append({"y_true": y_true_fold, "y_pred": y_pred_fold})

        # --- Plot example batches for this fold ---
        plotter(test_results, i) # plotter handles the plot_indices check

    # --- Aggregate and Print Final Results ---
    print("\n===== Cross-validation Complete! =====")

    if not all_fold_metrics:
        print("No folds were successfully completed.")
        return

    # Calculate mean and std deviation for each metric
    final_metrics = {}
    metric_keys = all_fold_metrics[0].keys()
    for key in metric_keys:
        # Handle potential NaN values when averaging
        values = [fold[key] for fold in all_fold_metrics if not np.isnan(fold[key])]
        if values: # Only calculate if there are non-NaN values
             mean_val = np.mean(values)
             std_val = np.std(values)
        else: # If all values were NaN
             mean_val = np.nan
             std_val = np.nan
        final_metrics[key] = {"mean": mean_val, "std": std_val}


    print("\n--- Final LOSO-CV Metrics (Mean ± Std) ---")
    for metric, values in final_metrics.items():
        print(f"{metric.capitalize():<12}: {values['mean']:.4f} ± {values['std']:.4f}")

    # Plot accuracy per fold
    accuracies = [fold["accuracy"] for fold in all_fold_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-')
    plt.xlabel("Fold (Left-Out Subject Index)")
    plt.ylabel("Accuracy")
    plt.title("Leave-One-Subject-Out Accuracy per Fold (Transformer)")
    plt.xticks(range(1, len(accuracies) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05) # Set Y axis limits
    plt.show()

    # Combined Confusion Matrix
    y_true_all = np.concatenate([fold["y_true"] for fold in all_fold_predictions])
    y_pred_all = np.concatenate([fold["y_pred"] for fold in all_fold_predictions])
    cm_combined = confusion_matrix(y_true_all, y_pred_all)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_combined, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Combined Confusion Matrix (All Folds - Transformer)')
    plt.show()

    print(f"{'='*20} Run Finished for: {npz_file_path} {'='*20}\n")


# In[5]: Execute Runs for Different Stage Classifications

if __name__ == "__main__":
    # --- Ensure the NPZ files exist in OUTPUT_DIR before running ---
    # You might need to uncomment the Data Loading/Preprocessing block above
    # if these files haven't been created yet.

    # --- Run 2-Stage Classification ---
    run(
        npz_file_path='BetterModels/2stage_sleep_dataset.npz',
        class_labels=["Awake", "Sleep"]
    )

    # --- Run 3-Stage Classification ---
    run(
        npz_file_path='BetterModels/3stage_sleep_dataset.npz',
        class_labels=["Awake", "REM", "NREM"]
    )

    # --- Run 4-Stage Classification ---
    run(
        npz_file_path='BetterModels/4stage_sleep_dataset.npz',
        class_labels=["Awake", "REM", "Light Sleep", "Deep Sleep"]
    )

    # # --- Run 6-Stage Classification ---
    # run(
    #     npz_file_path=os.path.join(OUTPUT_DIR, '6stage_sleep_dataset.npz'),
    #     class_labels=["Awake", "REM", "Stage 1/2", "Stage 3", "Stage 4", "Stage 5"] # Adjust labels if needed based on data
    #     # Note: Original UCDDB might use 0=W, 1=REM, 2=S1, 3=S2, 4=S3, 5=S4. Adjust labels accordingly.
    #     # Using: ["Awake", "REM", "S1", "S2", "S3", "S4"] might be more accurate representation of 0-5 stages.
    # )