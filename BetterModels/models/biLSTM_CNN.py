from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.signal import savgol_filter
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

input_size = 2
hidden_size = 64
num_layers = 1
batch_size = 16
num_epochs = 10
learning_rate = 0.0005


class SleepCNNLSTM(nn.Module):
    def __init__(self, n_features=5, n_classes=2, conv_channels=64, lstm_hidden=128, lstm_layers=3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(conv_channels, conv_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels*2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.bilstm = nn.LSTM(
            input_size=conv_channels*2, 
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2*2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        
        x = self.conv(x) 
        
        x = x.permute(0, 2, 1) 
        
        bilstm_out, (h_n, c_n) = self.bilstm(x)  
        

        last_timestep_output = bilstm_out[:, -1, :] 
        

        last_layer_forward = h_n[-2, :, :] 
        last_layer_backward = h_n[-1, :, :]  
        last_layer_combined = torch.cat([last_layer_forward, last_layer_backward], dim=1)
        
        combined_features = torch.cat(
            [last_timestep_output, last_layer_combined], 
            dim=1
        )  
        
        output = self.classifier(combined_features)
        return output

def normalize_channels(data):
    means = np.mean(data, axis=(0, 2))
    stds = np.std(data, axis=(0, 2))
    norm_data = (data - means[:, None]) / stds[:, None]
    return norm_data

def apply_sg_filter(data, window=25, order=5):
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered[i, j] = savgol_filter(data[i, j], window, order)
    return filtered

def restore_shape(data, original_shape):
    return data.reshape(-1, original_shape[1], original_shape[2])

def train_loop(model, num_epochs, criterion, optimizer, train_loader, device):
    train_results = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)  
            loss = criterion(outputs, labels)

            train_results.append((labels, (outputs)))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}", end='\r')

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return train_results

def test_loop(model, criterion, test_loader, device):
    
    test_results = []

    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # preds = torch.sigmoid(outputs)  
            preds = outputs  
            # probs = torch.softmax(outputs, dim=1)
            predicted_labels = torch.softmax(outputs, dim=1).argmax(dim=1)
            # predicted_labels = (preds > 0.5).float()  
            test_results.append((labels, preds))

            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}")
    return test_results

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve, f1_score, 
    precision_score, recall_score, roc_auc_score
)
from scipy.special import softmax

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- (Keep other functions like SleepCNNLSTM, normalize_channels, etc. as they are) ---
# --- (Keep train_loop and test_loop with the .squeeze(1) fix applied) ---

def calculate_multiclass_specificity(cm):
    """Calculates per-class and weighted specificity from a confusion matrix."""
    n_classes = cm.shape[0]
    per_class_specificity = []
    supports = []
    total_samples = cm.sum()

    for k in range(n_classes):
        # True Positives for class k
        tp = cm[k, k]
        # False Positives for class k (sum of column k minus TP)
        fp = cm[:, k].sum() - tp
        # False Negatives for class k (sum of row k minus TP)
        fn = cm[k, :].sum() - tp
        # True Negatives for class k (sum of all cells not in row k or col k)
        # Alternatively: Total samples - (TP + FP + FN)
        tn = total_samples - (tp + fp + fn)

        # Specificity for class k
        denominator = (tn + fp)
        spec_k = tn / denominator if denominator > 0 else 0.0
        per_class_specificity.append(spec_k)

        # Support for class k (number of true instances)
        support_k = cm[k, :].sum()
        supports.append(support_k)

    # Weighted Specificity
    weighted_specificity = np.sum(np.array(per_class_specificity) * np.array(supports)) / total_samples if total_samples > 0 else 0.0

    return per_class_specificity, weighted_specificity, supports


def load_metrics(test_results, i, class_labels=None): # Added class_labels optional arg
    # test_results is a list of (true_labels_batch, logits_batch) tuples
    y_true_batches = [batch[0] for batch in test_results]
    y_logits_batches = [batch[1] for batch in test_results]

    # Concatenate all batches
    y_true = torch.cat(y_true_batches).cpu().numpy()          # Shape: (n_samples,)
    y_logits = torch.cat(y_logits_batches)              # Shape: (n_samples, n_classes)
    y_prob = torch.softmax(y_logits, dim=1).cpu().numpy()     # Shape: (n_samples, n_classes)
    y_pred = np.argmax(y_prob, axis=1)                  # Shape: (n_samples,)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    print(f"--- Fold {i+1} Metrics ---")
    print("Confusion Matrix:\n", cm)

    # Initialize metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = np.nan
    recall = np.nan
    f1 = np.nan
    auc = np.nan
    specificity = np.nan # Will be overwritten by weighted average for multi-class
    per_class_specificity_list = None # Store per-class values

    # --- Binary Specific Metrics ---
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        print("tn", tn)
        print("fp", fp)
        print("fn", fn)
        print("tp", tp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"Specificity: {specificity:.4f}")
        # For binary AUC/PR, use probability of the positive class (class 1)
        y_prob_positive = y_prob[:, 1]
        try:
            auc = roc_auc_score(y_true, y_prob_positive)
            print(f"AUC-ROC: {auc:.4f}")
        except ValueError as e:
            print(f"Could not calculate binary AUC: {e}")
            auc = np.nan

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0) # Recall is Sensitivity
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Optional plots for specific folds
        if i in [0, 1, 15, 24]:
            # PR Curve
            try:
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob_positive)
                plt.figure(figsize=(6, 6))
                plt.plot(recall_vals, precision_vals, marker='.', label='PR Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Fold {i+1} Precision-Recall Curve')
                plt.legend()
                plt.grid()
                plt.show()
            except ValueError as e:
                print(f"Could not plot PR curve: {e}")

            # ROC Curve
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, marker='.', label=f'ROC curve (area = {auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Fold {i+1} ROC Curve')
                plt.legend(loc="lower right")
                plt.grid()
                plt.show()
            except ValueError as e:
                 print(f"Could not plot ROC curve: {e}")


    # --- Multi-class Metrics ---
    else:
        print("\nMulti-class classification report:")
        # Use class_labels if provided for better report readability
        target_names = class_labels if class_labels and len(class_labels) == n_classes else None
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

        # Calculate macro-averaged metrics (unweighted mean)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0) # Macro Recall/Sensitivity
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Calculate weighted-averaged metrics (weighted by support)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Calculate Specificity (per-class and weighted)
        per_class_specificity_list, weighted_specificity, supports = calculate_multiclass_specificity(cm)
        specificity = weighted_specificity # Store weighted average for the main return value

        print("\n--- Specificity Calculation ---")
        if target_names:
             for idx, spec in enumerate(per_class_specificity_list):
                 print(f"Specificity (Class: {target_names[idx]}): {spec:.4f}")
        else:
            for idx, spec in enumerate(per_class_specificity_list):
                 print(f"Specificity (Class: {idx}): {spec:.4f}")
        print(f"Weighted Average Specificity: {weighted_specificity:.4f}")
        print("-----------------------------")


        # AUC for multi-class needs OvR or OvO approach with probabilities
        try:
             # Needs probabilities per class (y_prob)
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro') # Macro OvR AUC
            auc_weighted = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted') # Weighted OvR AUC
            print(f"\nAUC-ROC (Macro OvR): {auc:.4f}")
            print(f"AUC-ROC (Weighted OvR): {auc_weighted:.4f}")
        except ValueError as e:
            print(f"\nCould not calculate multi-class AUC: {e}")
            auc = np.nan # Assign NaN if calculation fails

        print(f"\n--- Averaged Metrics ---")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall/Sensitivity (Macro): {recall:.4f}")
        print(f"F1-Score (Macro): {f1:.4f}")
        print(f"Precision (Weighted): {precision_weighted:.4f}")
        print(f"Recall/Sensitivity (Weighted): {recall_weighted:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print("------------------------")


    # --- Common Metrics & Plotting ---
    print(f"\nOverall Accuracy: {accuracy:.4f}")


    if i in [0, 1, 15, 24]:
        display_labels = class_labels if class_labels and len(class_labels) == n_classes else None
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Fold {i+1} Confusion Matrix')
        plt.show()

    # Return macro-averaged metrics by default, plus accuracy and weighted specificity
    # You could choose to return weighted averages instead if preferred
    return accuracy, precision, recall, specificity, f1, auc, y_true, y_pred # Return y_pred for consistency


def plotter(test_results):
    y_true = [i[0].cpu().numpy() for i in test_results]
    y_pred = [(i[1] > 0.5).cpu().numpy() for i in test_results]


    fig, ax = plt.subplots(2,2, figsize=(12, 6))

    ax[0,0].plot(y_true[0] , color='steelblue', linewidth=1.5, label='Truth')
    ax[0,0].plot(y_pred[0], color='red', linewidth=1.5, label='Prediction', alpha=0.5)
    ax[0, 0].set_xlabel('Batch No', fontsize=12)
    ax[0, 0].set_ylabel('Sleep Stage', fontsize=12)
    ax[0,0].legend()

    ax[0,1].plot(y_true[1], color='steelblue', linewidth=1.5, label='Truth')
    ax[0,1].plot(y_pred[1], color='red', linewidth=1.5, label='Prediction', alpha=0.5)
    ax[0, 1].set_xlabel('Batch No', fontsize=12)
    ax[0, 1].set_ylabel('Sleep Stage', fontsize=12)
    ax[0,1].legend()

    ax[1,0].plot(y_true[2], color='steelblue', linewidth=1.5, label='Truth')
    ax[1,0].plot(y_pred[2], color='red', linewidth=1.5, label='Prediction', alpha=0.5)
    ax[1, 0].set_xlabel('Batch No', fontsize=12)
    ax[1, 0].set_ylabel('Sleep Stage', fontsize=12)
    ax[1,0].legend()

    ax[1,1].plot(y_true[15], color='steelblue', linewidth=1.5, label='Truth')
    ax[1,1].plot(y_pred[15], color='red', linewidth=1.5, label='Prediction', alpha=0.5)
    ax[1, 1].set_xlabel('Batch No', fontsize=12)
    ax[1, 1].set_ylabel('Sleep Stage', fontsize=12)
    ax[1,1].legend()

    fig.tight_layout()
    plt.show()


def run(loaded_data, class_labels=["Awake", "Sleep"]):
    # Load the dataset
    X_list_loaded = [loaded_data[f'X_{i}'] for i in range(25)]
    Y_list_loaded = [loaded_data[f'Y_{i}'] for i in range(25)]
    dataset = [(X_list_loaded[i], Y_list_loaded[i]) for i in range(25)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fold_results = []
    fold = []
    criterioncnnlstm = nn.CrossEntropyLoss()
    results = []
    # Cross-validation loop
    for i, data in enumerate(dataset):
        print(f"Leave-One-Subject-Out CV - Subject: {i+1}\n-----------------------------------")
        
        # Split data: leave out the current subject for testing
        train_data = [x for j, x in enumerate(dataset) if j != i]
        test_data = data

        # Concatenate features and labels for training
        train_features = np.vstack([x[0] for x in train_data])
        train_labels = np.vstack([x[1] for x in train_data])
        test_features = test_data[0]
        test_labels = test_data[1]

        # Preprocess the data
        train_norm = normalize_channels(train_features)
        train_sg = apply_sg_filter(train_norm)
        test_norm = normalize_channels(test_features)
        test_sg = apply_sg_filter(test_norm)


        # Convert to PyTorch tensors
        processed_X_train_tensor = torch.tensor(train_sg, dtype=torch.float32)
        processed_y_train_tensor = torch.tensor(train_labels, dtype=torch.long).view(-1)
        processed_X_test_tensor = torch.tensor(test_sg, dtype=torch.float32)
        processed_y_test_tensor = torch.tensor(test_labels, dtype=torch.long).view(-1) 

        # Create DataLoaders
        train_dataset = list(zip(processed_X_train_tensor, processed_y_train_tensor))
        test_dataset = list(zip(processed_X_test_tensor, processed_y_test_tensor))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Reinitialize the model for each fold
        modelcnnlstm = SleepCNNLSTM(n_features=5, n_classes=len(class_labels), conv_channels=64, lstm_hidden=128, lstm_layers=3).to(device)
        optimizercnnlstm = optim.Adam(modelcnnlstm.parameters(), lr=learning_rate)


        # Train the model
        # for t in range(num_epochs):
        #     print(f"Epoch {t+1}\n-------------------------------")
        train_loop(num_epochs=num_epochs, train_loader=train_loader, model=modelcnnlstm, criterion=criterioncnnlstm, optimizer=optimizercnnlstm, device=device)

        # Evaluate on the test subject
        test_results = test_loop(test_loader=test_loader, model=modelcnnlstm, criterion=criterioncnnlstm, device=device)
        results.append((i, test_results))

        # Process metrics and plot results for each fold
        y_true = [i[0] for i in test_results]
        y_prob = [i[1] for i in test_results]


        accuracy_score, precision, recall, specificity, f1, auc, y_true, y_prob = load_metrics(test_results, i, class_labels=class_labels)
        # Compute metrics for this fold
        fold_metrics = {
            "accuracy": accuracy_score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "specificity": specificity,
        }

        fold_results.append(fold_metrics)

        fold_pred = {
            "y_true": y_true,
            "y_pred": (y_prob > 0.5).astype(int)
        }
        fold.append(fold_pred)

        print(f"Fold {i+1} metrics: {fold_metrics}")
        
        if i in [0, 1, 15, 24]:
        # load_metrics(test_results)
            plotter(test_results)

    print("Cross-validation complete!")

    final_metrics = {metric: {"mean": np.mean([fold[metric] for fold in fold_results]),
                            "std": np.std([fold[metric] for fold in fold_results])}
                    for metric in fold_results[0]}

    print("\nFinal Metrics:")
    for metric, values in final_metrics.items():
        print(f"{metric.capitalize()} - Mean: {values['mean']:.4f}, Std: {values['std']:.4f}")

    # Plot accuracy for each fold
    accuracies = [fold["accuracy"] for fold in fold_results]
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.xlabel("Fold (Subject)")
    plt.ylabel("Accuracy")
    plt.title("Leave-One-Subject-Out Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Optional: Confusion Matrix (combine all folds)
    # Optional: Confusion Matrix (combine all folds) with class labels
    y_true_all = np.concatenate([fold["y_true"] for fold in fold])
    y_pred_all = np.concatenate([fold["y_pred"] for fold in fold])
    cm = confusion_matrix(y_true_all, y_pred_all)

    # Create display with your class names
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=class_labels
    )
    disp.plot(cmap=plt.cm.Blues)          # you can still choose your colormap
    disp.ax_.set_xlabel('Predicted Label')
    disp.ax_.set_ylabel('True Label')
    plt.title('Combined Confusion Matrix')
    plt.show()

# loaded_data = np.load('BetterModels/2stage_sleep_dataset.npz', allow_pickle=True)
# run(loaded_data, class_labels=["Awake", "Sleep"])

# loaded_data = np.load('BetterModels/3stage_sleep_dataset.npz', allow_pickle=True)
# run(loaded_data, class_labels=["Awake", "REM", "NREM"])

loaded_data = np.load('BetterModels/4stage_sleep_dataset.npz', allow_pickle=True)
run(loaded_data, class_labels=["Awake", "REM", "Light Sleep", "Deep Sleep"])

loaded_data = np.load('BetterModels/sleep_dataset.npz', allow_pickle=True)
run(loaded_data, class_labels=["Awake", "REM", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]) 