import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier # Remove XGBoost import
from sklearn.ensemble import RandomForestClassifier # Import RandomForest
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import tsfel
from joblib import Memory
from sklearn.model_selection import LeaveOneGroupOut
import warnings # Optional: To suppress potential warnings if needed

# --- Data Loading (Your existing code) ---



def run(loaded, class_labels=None):
    # Explicitly check keys assuming 25 subjects for X and Y
    num_subjects = 25 # Or determine dynamically if needed

    X_list_loaded = [loaded[f'X_{i}'] for i in range(num_subjects)]
    Y_list_loaded = [loaded[f'Y_{i}'] for i in range(num_subjects)]
    dataset = [(X_list_loaded[i], Y_list_loaded[i]) for i in range(num_subjects)]


    print("-------------------------------------------------------------------------------------------")
    print("Shape of signals (first subject):", dataset[0][0].shape)
    print("Shape of sleep stages (first subject):", dataset[0][1].shape)


    for i in range(25):
        print(dataset[i][0].shape, type(dataset[i][0]))
        signals, labels = dataset[i]  # unpack the tuple
        print(signals.shape, type(signals))

        # # Only reshape if signal is 3D
        # signals = signals[:, [0,1], :]  # reduce 3D to 2D
        # signals = signals.reshape(signals.shape[0], 1, signals.shape[1])  # reshape to 2D (n_samples, n_features)
        # # Repack as a new tuple
        # dataset[i] = (signals, labels)
        print(dataset[i][0].shape, type(signals))
            

    print("Shape of signals (first subject):", dataset[0][0].shape)

    memory = Memory(location='./tsfel_cache_xgboost', verbose=0) 

    @memory.cache
    def extract_tsfel_features(data_i, fs):
        cfg = tsfel.get_features_by_domain()
        print(f"Extracting features for a segment of shape {data_i.shape}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            features = tsfel.time_series_features_extractor(cfg, data_i, fs=fs, verbose=0)
        return features

    features_list = []
    y_list = []
    groups_list = []

    print("\nStarting feature extraction for all subjects...")
    for subj_idx, (X, y) in enumerate(dataset):
        print(f"Processing Subject {subj_idx}...")
        n_i, C, T = X.shape
        data_i = np.transpose(X, (0, 2, 1))

        # Convert data to float64 BEFORE TSFEL, as it expects floats
        data_i_float = data_i.astype(np.float64)

        feats_i = extract_tsfel_features(data_i_float, fs=8)

        features_list.append(feats_i)
        y_flat = y.ravel().astype(int)
        y_list.append(y_flat)
        groups_list.append(np.full(n_i, subj_idx))

    print("Feature extraction complete.")

    
    features_all = pd.concat(features_list, axis=0, ignore_index=True) # Use ignore_index
    y_all      = np.concatenate(y_list)
    groups_all = np.concatenate(groups_list)

    unique_labels = np.unique(y_all)
    print(f"\nUnique labels found in aggregated data: {unique_labels}")
    class_labels_numeric = np.arange(len(unique_labels))

    def calculate_specificity(conf_matrix, class_labels):
        specificities = {}
        n_classes = conf_matrix.shape[0]
        total_sum = conf_matrix.sum()
        for i in range(n_classes):
            class_label = class_labels[i] # Get the actual label (0 or 1)
            TP = conf_matrix[i, i]
            FP = conf_matrix[:, i].sum() - TP
            FN = conf_matrix[i, :].sum() - TP
            TN = total_sum - TP - FP - FN
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            specificities[class_label] = specificity # Use the actual label as key
        return specificities

    logo = LeaveOneGroupOut()
    accuracies = []
    all_y_true = []
    all_y_pred = []

    # class_labels = np.array([0, 1]) 
    


    print("\nStarting Leave-One-Group-Out Cross-Validation with XGboost...")
    n_splits = logo.get_n_splits(groups=groups_all)

    for fold_num, (train_idx, test_idx) in enumerate(logo.split(features_all, y_all, groups_all)):
        # 4a) Split
        # Use .values to get numpy arrays, which is generally safer for sklearn
        X_train = features_all.iloc[train_idx].values
        y_train = y_all[train_idx]
        X_test  = features_all.iloc[test_idx].values
        y_test  = y_all[test_idx]
        current_subj = groups_all[test_idx[0]]

      
        # 4b) Scale
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)

    
        clf = XGBClassifier(n_estimators=100, # Number of trees
                             random_state=42,   # For reproducibility
                             n_jobs=-1,         # Use all available CPU cores
                             )

        clf.fit(X_train_s, y_train)
        # 4d) Predict & score
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Calculate confusion matrix for the fold using NUMERIC labels
        # Ensure labels parameter uses the numeric ones for correct indexing
        cm_fold = confusion_matrix(y_test, y_pred, labels=class_labels_numeric)

        # Calculate specificity for the fold (using numeric labels for calculation)
        specificity_fold = calculate_specificity(cm_fold, class_labels_numeric)

        print(f"Left-out subject {current_subj} (Fold {fold_num+1}/{n_splits}): Accuracy = {acc:.3f}")
        # Print specificity per class for the fold, using the numeric labels from the spec dict keys
        spec_str = ", ".join([f"Class {int(k)} ({class_labels_numeric[int(k)]}): {v:.3f}" for k, v in specificity_fold.items()])
        print(f"  Fold Specificity: {spec_str}")

        # Store predictions and true values for overall metrics
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # --- Overall Performance & Metrics ---
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\nMean accuracy (XGboost): {mean_acc:.3f} ± {std_acc:.3f}")

    # Calculate overall confusion matrix using NUMERIC labels
    overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=class_labels_numeric)

    # Calculate overall specificity (using numeric labels for calculation)
    overall_specificity = calculate_specificity(overall_cm, class_labels_numeric)
    print("\nOverall Specificity per Class:")
    for label_num, spec in overall_specificity.items():
        # Map numeric label back to display label for printing
        display_label = class_labels[int(label_num)]
        print(f"  Class {int(label_num)} ({display_label}): {spec:.3f}")

    # --- Plot Overall Confusion Matrix ---
    print("\nPlotting Overall Confusion Matrix...")
    plt.figure(figsize=(8, 6)) # Adjusted size slightly for 2x2 matrix
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
                # Use DISPLAY labels for the plot axes
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Overall Confusion Matrix (XGboost)\nMean Acc: {mean_acc:.3f}')
    plt.tight_layout() # Adjust layout
    plt.show()

    print("\nScript finished.")


# loaded_data = np.load('BetterModels/2stage_sleep_dataset_essential.npz', allow_pickle=True)
# run(loaded_data, class_labels=["Awake", "Sleep"])

loaded_data = np.load('BetterModels/3stage_sleep_dataset_essential.npz', allow_pickle=True)
run(loaded_data, class_labels=["Awake", "REM", "NREM"])

loaded_data = np.load('BetterModels/4stage_sleep_dataset_essential.npz', allow_pickle=True)
run(loaded_data, class_labels=["Awake", "REM", "Light Sleep", "Deep Sleep"])

loaded_data = np.load('BetterModels/sleep_dataset_essential.npz', allow_pickle=True)
run(loaded_data, class_labels=["Awake", "REM", "Stage 1", "Stage 2", "Stage 3", "Stage 4"])