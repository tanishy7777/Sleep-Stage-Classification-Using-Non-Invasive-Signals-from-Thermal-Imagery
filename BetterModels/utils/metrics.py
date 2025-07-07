
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    classification_report, precision_recall_curve, f1_score, 
    precision_score, recall_score, roc_auc_score
)
from scipy.special import softmax
import torch
import matplotlib.pyplot as plt

def load_metrics(test_results, fold_index):
    # Extract y_true and y_prob lists from test_results.
    y_true_list = [item[0] for item in test_results]
    y_prob_list = [item[1] for item in test_results]

    # Concatenate the list of tensors into one tensor.
    y_prob_tensor = torch.cat([
        torch.tensor(prob, dtype=torch.float32) if not isinstance(prob, torch.Tensor) else prob 
        for prob in y_prob_list
    ])
    
    # Use y_prob_tensor directly to get predicted labels.
    y_pred = (y_prob_tensor.cpu().detach() > 0.5).float()
    
    # Concatenate ground truth tensors.
    y_true_tensor = torch.cat(y_true_list)
    
    # Convert tensors to numpy arrays for metric computations.
    y_true_np = y_true_tensor.cpu().detach().numpy()
    y_prob_np = y_prob_tensor.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    
    # Compute the confusion matrix.
    cm = confusion_matrix(y_true_np, y_pred_np)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    print("tn:", tn)
    print("fp:", fp)
    print("fn:", fn)
    print("tp:", tp)
    print("Confusion Matrix:\n", cm)
    
    if fold_index in [0, 1, 15, 24]:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()
    
    # Compute additional metrics.
    precision_val = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall_val = recall_score(y_true_np, y_pred_np, zero_division=0)
    specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0  
    f1_val = f1_score(y_true_np, y_pred_np, zero_division=0)
    try:
        auc_val = roc_auc_score(y_true_np, y_prob_np)
    except ValueError:
        auc_val = 0.0
    
    print(f"Precision: {precision_val:.4f}")
    print(f"Recall (Sensitivity): {recall_val:.4f}")
    print(f"Specificity: {specificity_val:.4f}")
    print(f"F1-Score: {f1_val:.4f}")
    print(f"AUC-ROC: {auc_val:.4f}")
    
    if fold_index in [0, 1, 15, 24]:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_np, y_prob_np)
        plt.figure(figsize=(6, 6))
        plt.plot(recall_vals, precision_vals, marker='.', label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid()
        plt.show()
    
    correct_predictions = (y_true_np == y_pred_np).sum()
    total_predictions = len(y_true_np)
    accuracy = correct_predictions / total_predictions

    return accuracy, precision_val, recall_val, specificity_val, f1_val, auc_val, y_true_np, y_prob_np