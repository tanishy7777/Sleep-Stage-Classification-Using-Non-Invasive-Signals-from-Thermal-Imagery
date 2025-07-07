import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier       # or swap out for RandomForest below
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import tsfel
from joblib import Memory
from sklearn.model_selection import LeaveOneGroupOut
import warnings

# new import for undersampling
from imblearn.under_sampling import RandomUnderSampler  

def run(loaded, class_labels=None):
    # Dynamically get number of subjects
    num_subjects = len([k for k in loaded if k.startswith('X_')])

    X_list_loaded = [loaded[f'X_{i}'] for i in range(num_subjects)]
    Y_list_loaded = [loaded[f'Y_{i}'] for i in range(num_subjects)]
    dataset = list(zip(X_list_loaded, Y_list_loaded))

    print("-------------------------------------------------------------------------------------------")
    print("Shape of signals (first subject):", dataset[0][0].shape)
    print("Shape of sleep stages (first subject):", dataset[0][1].shape)

    # build group labels
    groups_list = [np.full(X.shape[0], subj_idx)
                   for subj_idx, (X, _) in enumerate(dataset)]

    # aggregate into DataFrame / arrays
    features_all = pd.concat(
        [pd.DataFrame(X) for X, _ in dataset], axis=0, ignore_index=True
    )
    y_all      = np.concatenate([Y.ravel().astype(int) for _, Y in dataset])
    groups_all = np.concatenate(groups_list)

    unique_labels = np.unique(y_all)
    print(f"\nUnique labels found in aggregated data: {unique_labels}")
    class_labels_numeric = np.arange(len(unique_labels))

    def calculate_specificity(conf_matrix, class_labels_idx):
        specificities = {}
        total = conf_matrix.sum()
        for i in range(conf_matrix.shape[0]):
            TP = conf_matrix[i, i]
            FP = conf_matrix[:, i].sum() - TP
            TN = total - TP - FP - (conf_matrix[i, :].sum() - TP)
            specificities[class_labels_idx[i]] = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        return specificities

    logo = LeaveOneGroupOut()
    accuracies, all_y_true, all_y_pred = [], [], []

    # print("\nStarting Leave-One-Group-Out Cross-Validation with undersampling...")
    n_splits = logo.get_n_splits(groups=groups_all)
    print(f"\nStarting Leave-One-Group-Out Cross-Validation with undersampling "
       f"({n_splits} folds)...")
    for fold_num, (train_idx, test_idx) in enumerate(
        logo.split(features_all, y_all, groups=groups_all)
    ):
        # split
        X_train = features_all.iloc[train_idx].values
        y_train = y_all[train_idx]
        X_test  = features_all.iloc[test_idx].values
        y_test  = y_all[test_idx]
        subj_out = groups_all[test_idx[0]]

        # 1) undersample majority class in training data
        rus = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

        # # 2) scale
        # scaler = StandardScaler().fit(X_train_res)
        # X_train_s = scaler.transform(X_train_res)
        # X_test_s  = scaler.transform(X_test)

        # 2) scale (suppress runtime warnings from constant‐feature divides)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scaler = StandardScaler().fit(X_train_res)
            X_train_s = scaler.transform(X_train_res)
            X_test_s  = scaler.transform(X_test)

        # 3) fit classifier (here XGBoost, swap in RandomForest if you like)
        clf = XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        # clf = RandomForestClassifier(n_estimators=100, class_weight=None, random_state=42, n_jobs=-1)

        clf.fit(X_train_s, y_train_res)

        # 4) predict & score
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        cm_fold = confusion_matrix(y_test, y_pred, labels=class_labels_numeric)
        spec_fold = calculate_specificity(cm_fold, class_labels_numeric)

        spec_str = ", ".join(
            f"Class {lbl}: {spec_fold[idx]:.3f}"
            for idx, lbl in enumerate(class_labels)
        )
        # print(f"Left-out subject {subj_out} (Fold {fold_num+1}/{logo.get_n_splits()}): "
        #       f"Accuracy = {acc:.3f}")
        print(f"Left-out subject {subj_out} (Fold {fold_num+1}/{n_splits}): "
              f"Accuracy = {acc:.3f}")
        print(f"  Fold Specificity: {spec_str}")

    # overall
    mean_acc = np.mean(accuracies)
    std_acc  = np.std(accuracies)
    print(f"\nMean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    overall_cm  = confusion_matrix(all_y_true, all_y_pred, labels=class_labels_numeric)
    overall_spec = calculate_specificity(overall_cm, class_labels_numeric)

    print("\nOverall Specificity per Class:")
    for idx, lbl in enumerate(class_labels):
        print(f"  {lbl}: {overall_spec[idx]:.3f}")

    # plot
    plt.figure(figsize=(8,6))
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Mean Acc: {mean_acc:.3f})')
    plt.tight_layout()
    plt.show()

# Usage:
# loaded_data = np.load('BetterModels/2_stage_sleep_features.npz', allow_pickle=True)
# run(loaded_data, class_labels=["Awake", "Sleep"])

loaded_data = np.load('BetterModels/3_stage_sleep_features.npz', allow_pickle=True)
run(loaded_data, class_labels=["Awake", "REM", "NREM"])

loaded_data = np.load('BetterModels/4_stage_sleep_features.npz', allow_pickle=True)
run(loaded_data, class_labels=["Awake", "REM", "Light Sleep", "Deep Sleep"])

