from utils.traintest import train_loop, test_loop
from utils.metrics import load_metrics
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import savgol_filter

def apply_sg_filter(data, window=25, order=5):
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered[i, j] = savgol_filter(data[i, j], window, order)
    return filtered

def cross_validation(dataset, modelClass, apply_sgfilter, num_epochs, batch_size, learning_rate, **model_params):

    for i, data in enumerate(dataset):
        print(f"Leave-One-Subject-Out CV - Subject: {i+1}\n-----------------------------------")
        
        train_data = [x for j, x in enumerate(dataset) if j != i]
        test_data = data

        X_train = np.vstack([x[0] for x in train_data])
        Y_train = np.vstack([x[1] for x in train_data])
        X_test = test_data[0]
        Y_test = test_data[1]

        print(f"Train data shape: {X_train.shape}, Train labels shape: {Y_train.shape}")
        print(f"Test data shape: {X_test.shape}, Test labels shape: {Y_test.shape}")
        if apply_sgfilter:
            # TODO: Apply Savitzky-Golay filter across all channels independently
            X_train = apply_sg_filter(X_train)
            X_test = apply_sg_filter(X_test)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)

        # # Create sequence datasets (each sample is now a sequence of epochs)
        # train_dataset = SleepSequenceDataset(X_train, Y_train, seq_length=seq_length)
        # test_dataset = SleepSequenceDataset(X_test, Y_test, seq_length=seq_length)

        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Reinitialize the model for this fold
        model = modelClass(n_channels=n_channels, 
                                        epoch_samples=epoch_samples, 
                                        cnn_out_features=cnn_out_features, 
                                        lstm_hidden=lstm_hidden, 
                                        lstm_layers=lstm_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_loop(num_epochs=num_epochs, 
                train_loader=train_loader, 
                model=model, 
                criterion=criterion, 
                optimizer=optimizer, 
                device=device)

        # Evaluate on the test subject
        test_results = test_loop(test_loader=test_loader, 
                                model=model, 
                                criterion=criterion, 
                                device=device)
        results.append((i, test_results))

        # Process metrics and plot results for each fold
        y_true = [result[0] for result in test_results]
        y_prob = [result[1] for result in test_results]

        accuracy_score, precision, recall, specificity, f1, auc, y_true, y_prob = load_metrics(test_results, i)
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
            plotter(test_results)

    print("Cross-validation complete!")

    #############################
    # Final Metrics & Plots
    #############################

    final_metrics = {metric: {"mean": np.mean([fmetrics[metric] for fmetrics in fold_results]),
                            "std": np.std([fmetrics[metric] for fmetrics in fold_results])}
                    for metric in fold_results[0]}

    print("\nFinal Metrics:")
    for metric, values in final_metrics.items():
        print(f"{metric.capitalize()} - Mean: {values['mean']:.4f}, Std: {values['std']:.4f}")

    # Plot accuracy for each fold
    accuracies = [fmetrics["accuracy"] for fmetrics in fold_results]
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.xlabel("Fold (Subject)")
    plt.ylabel("Accuracy")
    plt.title("Leave-One-Subject-Out Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Optional: Combined Confusion Matrix from all folds
    y_true_all = np.concatenate([f["y_true"] for f in fold])
    y_pred_all = np.concatenate([f["y_pred"] for f in fold])
    cm = confusion_matrix(y_true_all, y_pred_all)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()



import numpy as np
import pandas as pd
import tsfel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier



features_list = []
n_channels = X_train.shape[1]

data = np.transpose(X_train, (0, 2, 1))
features = tsfel.time_series_features_extractor(cfg, data, fs=8)
features_list.append(features)
tsfel_x_train = pd.concat(features_list, axis=1)
print(tsfel_x_train.shape)


# 4) Set up leave‑one‑subject‑out CV
logo = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(features_df, features_df['y'], features_df['subject']):
    # Split features, labels
    X_train = features_df.iloc[train_idx].drop(['y','subject'], axis=1).values
    y_train = features_df.iloc[train_idx]['y'].values
    X_test  = features_df.iloc[test_idx ].drop(['y','subject'], axis=1).values
    y_test  = features_df.iloc[test_idx ]['y'].values

    # 5) Fit scaler **only on train**, then transform both
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 6) Train your classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_scaled, y_train)
    score = clf.score(X_test_scaled, y_test)
    print(f"Left‑out subject {features_df['subject'].iloc[test_idx[0]]}: accuracy = {score:.3f}")
