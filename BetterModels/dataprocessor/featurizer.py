import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch, savgol_filter
from scipy.stats import skew, kurtosis
from scipy.integrate import simpson
import antropy
import warnings

# Suppress specific warnings if needed (e.g., from entropy calculation on short/flat signals)
warnings.filterwarnings("ignore", message="Found duplicate knots")
warnings.filterwarnings("ignore", category=RuntimeWarning) # Suppress common numpy warnings like division by zero

# --- Helper Functions (Mostly unchanged) ---

def _calculate_signal_stats(signal):
    stats = {}
    # Ensure signal is not empty or all NaN
    if signal is None or signal.size == 0 or np.all(np.isnan(signal)):
        keys = ['mean','std','min','max','ptp','skew','kurt','median','p5','p95','iqr','variance','rms','auc']
        return {k: np.nan for k in keys}

    signal = signal[~np.isnan(signal)] # Remove NaNs for calculation
    if signal.size == 0:
        keys = ['mean','std','min','max','ptp','skew','kurt','median','p5','p95','iqr','variance','rms','auc']
        return {k: np.nan for k in keys}


    stats['mean'] = np.mean(signal)
    stats['std'] = np.std(signal)
    stats['min'] = np.min(signal)
    stats['max'] = np.max(signal)
    stats['ptp'] = np.ptp(signal)
    stats['skew'] = skew(signal) if stats['std'] > 1e-9 else 0.0
    stats['kurt'] = kurtosis(signal) if stats['std'] > 1e-9 else 0.0
    stats['median'] = np.median(signal)
    stats['p5'] = np.percentile(signal, 5)
    stats['p95'] = np.percentile(signal, 95)
    stats['iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
    stats['variance'] = np.var(signal)
    stats['rms'] = np.sqrt(np.mean(signal**2))
    try:
        stats['auc'] = simpson(np.abs(signal))
    except ValueError:
        stats['auc'] = np.nan # Handle potential integration errors
    return stats

def _calculate_rrv_features(peak_intervals_sec):
    rrv = {}
    keys = ['mean_ibi', 'std_ibi', 'rmssd', 'sdsd', 'cv']
    if peak_intervals_sec is None or len(peak_intervals_sec) < 2:
        return {k: np.nan for k in keys}

    rrv['mean_ibi'] = np.mean(peak_intervals_sec)
    rrv['std_ibi'] = np.std(peak_intervals_sec)

    successive_diffs = np.diff(peak_intervals_sec)
    if len(successive_diffs) > 0:
        rrv['rmssd'] = np.sqrt(np.mean(successive_diffs**2))
        rrv['sdsd'] = np.std(successive_diffs)
    else:
        rrv['rmssd'] = np.nan
        rrv['sdsd'] = np.nan

    if rrv['mean_ibi'] is not None and rrv['mean_ibi'] != 0 and not np.isnan(rrv['mean_ibi']):
        rrv['cv'] = rrv['std_ibi'] / rrv['mean_ibi']
    else:
        rrv['cv'] = np.nan

    return rrv

def _calculate_entropy_features(signal):
    entropy = {}
    if signal is None or len(signal) < 10: # Need sufficient length for entropy
         entropy['sample_entropy'] = np.nan
         entropy['perm_entropy'] = np.nan
         return entropy
    try:
        # Using m=2, r=0.2*std as common defaults for sample entropy
        r = 0.2 * np.std(signal)
        entropy['sample_entropy'] = antropy.sample_entropy(signal, order=2, metric='chebyshev', approximate=False, radius=r)
    except (ValueError, TypeError):
        entropy['sample_entropy'] = np.nan

    try:
        entropy['perm_entropy'] = antropy.perm_entropy(signal, order=3, normalize=True)
    except (ValueError, TypeError):
        entropy['perm_entropy'] = np.nan

    return entropy

# --- Feature Extraction Functions (Adapted for Robustness) ---

def extract_respiratory_features(signal, fs, prefix, peak_prominence_ratio=0.1, peak_distance_sec=1.0):
    """Extracts features from a respiratory signal (Nasal, Ribcage, or Abdomen)."""
    features = {}
    if signal is None or len(signal) < int(fs * 2): # Need at least a couple of seconds
        # Return NaNs if signal is too short or None
        stats = _calculate_signal_stats(None)
        features.update({f"{prefix}{k}": v for k, v in stats.items()})
        features[f'{prefix}num_breaths'] = 0
        features[f'{prefix}rr_mean'] = np.nan
        features[f'{prefix}rr_std'] = np.nan
        rrv_features = _calculate_rrv_features(None)
        features.update({f"{prefix}rrv_{k}": v for k, v in rrv_features.items()})
        features[f'{prefix}breath_amp_mean'] = np.nan
        features[f'{prefix}breath_amp_std'] = np.nan
        features[f'{prefix}breath_amp_max'] = np.nan
        entropy = _calculate_entropy_features(None)
        features.update({f"{prefix}entropy_{k}": v for k, v in entropy.items()})
        return features


    # Preprocessing (adjust window for 8Hz)
    try:
        win_len = min(int(fs * 0.5) | 1, len(signal) - 1 if len(signal) % 2 == 0 else len(signal))
        if win_len >= 3:
             signal_filt = savgol_filter(signal, window_length=win_len, polyorder=2)
        else:
             signal_filt = signal
    except ValueError:
        signal_filt = signal

    stats = _calculate_signal_stats(signal_filt)
    features.update({f"{prefix}{k}": v for k, v in stats.items()})

    peak_distance_samples = max(1, int(peak_distance_sec * fs)) # Ensure distance is at least 1
    peaks = np.array([]) # Initialize
    properties = {}
    try:
        # More robust prominence calculation
        signal_range = np.ptp(signal_filt) if not np.all(np.isnan(signal_filt)) else 0
        if signal_range > 1e-6:
            prominence = signal_range * peak_prominence_ratio
            # Adjust height based on median - might help find peaks around baseline
            height_threshold = np.median(signal_filt)
            peaks, properties = find_peaks(signal_filt,
                                         prominence=prominence,
                                         distance=peak_distance_samples,
                                         height=height_threshold) # Added height
        else: # Handle flat or near-flat signals
            peaks = np.array([])
            properties = {}

    except (ValueError, TypeError):
        peaks = np.array([])
        properties = {}


    if len(peaks) > 1:
        peak_intervals_samples = np.diff(peaks)
        peak_intervals_sec = peak_intervals_samples / fs
        features[f'{prefix}num_breaths'] = len(peaks)
        # Calculate RR only if there are valid intervals
        valid_intervals = peak_intervals_sec[peak_intervals_sec > 1e-6] # Avoid division by zero
        if len(valid_intervals) > 0:
            features[f'{prefix}rr_mean'] = 60.0 / np.mean(valid_intervals)
            features[f'{prefix}rr_std'] = np.std(60.0 / valid_intervals)
            rrv_features = _calculate_rrv_features(valid_intervals)
            features.update({f"{prefix}rrv_{k}": v for k, v in rrv_features.items()})
        else:
            features[f'{prefix}rr_mean'] = np.nan
            features[f'{prefix}rr_std'] = np.nan
            rrv_features = _calculate_rrv_features(None)
            features.update({f"{prefix}rrv_{k}": v for k, v in rrv_features.items()})

    else:
        features[f'{prefix}num_breaths'] = len(peaks)
        features[f'{prefix}rr_mean'] = 0 if len(peaks) == 1 else np.nan
        features[f'{prefix}rr_std'] = 0 if len(peaks) == 1 else np.nan
        rrv_features = _calculate_rrv_features(None)
        features.update({f"{prefix}rrv_{k}": v for k, v in rrv_features.items()})

    if len(peaks) > 0 :
        # Using filtered signal heights, potentially more robust than raw
        breath_amplitudes = signal_filt[peaks]
        features[f'{prefix}breath_amp_mean'] = np.mean(breath_amplitudes)
        features[f'{prefix}breath_amp_std'] = np.std(breath_amplitudes)
        features[f'{prefix}breath_amp_max'] = np.max(breath_amplitudes)
    else:
        features[f'{prefix}breath_amp_mean'] = np.nan
        features[f'{prefix}breath_amp_std'] = np.nan
        features[f'{prefix}breath_amp_max'] = np.nan

    entropy = _calculate_entropy_features(signal_filt)
    features.update({f"{prefix}entropy_{k}": v for k, v in entropy.items()})

    return features

def extract_thoracoabdominal_features(rib_signal, abd_signal, fs):
    """Extracts features related to synchronization between ribcage and abdomen."""
    features = {}
    prefix = 'sync_'

    if rib_signal is None or abd_signal is None or len(rib_signal) != len(abd_signal) or len(rib_signal) < int(fs * 2):
        # Return NaNs if signals are mismatched, too short, or None
        features[f'{prefix}max_xcorr'] = np.nan
        features[f'{prefix}lag_at_max_xcorr_samples'] = np.nan
        features[f'{prefix}lag_at_max_xcorr_sec'] = np.nan
        features[f'{prefix}amp_ratio_rib_abd'] = np.nan
        # features[f'{prefix}inst_phase_diff_mean'] = np.nan # If adding phase diff
        # features[f'{prefix}inst_phase_diff_std'] = np.nan
        return features

    # Preprocessing
    try:
        win_len = min(int(fs * 0.5) | 1, len(rib_signal) - 1 if len(rib_signal) % 2 == 0 else len(rib_signal))
        if win_len >= 3:
             rib_filt = savgol_filter(rib_signal, window_length=win_len, polyorder=2)
             abd_filt = savgol_filter(abd_signal, window_length=win_len, polyorder=2)
        else:
             rib_filt = rib_signal
             abd_filt = abd_signal
    except ValueError:
        rib_filt = rib_signal
        abd_filt = abd_signal

    # Normalize (handle potential zero std dev)
    rib_std = np.std(rib_filt)
    abd_std = np.std(abd_filt)
    rib_norm = (rib_filt - np.mean(rib_filt)) / (rib_std + 1e-9)
    abd_norm = (abd_filt - np.mean(abd_filt)) / (abd_std + 1e-9)

    # Cross-Correlation
    try:
        correlation = np.correlate(rib_norm, abd_norm, mode='full')
        lags = np.arange(-len(abd_norm) + 1, len(rib_norm))
        max_corr_idx = np.argmax(np.abs(correlation))
        features[f'{prefix}max_xcorr'] = correlation[max_corr_idx]
        features[f'{prefix}lag_at_max_xcorr_samples'] = lags[max_corr_idx]
        features[f'{prefix}lag_at_max_xcorr_sec'] = lags[max_corr_idx] / fs
    except (ValueError, TypeError):
        features[f'{prefix}max_xcorr'] = np.nan
        features[f'{prefix}lag_at_max_xcorr_samples'] = np.nan
        features[f'{prefix}lag_at_max_xcorr_sec'] = np.nan

    # Amplitude Ratio
    if abd_std > 1e-9:
        features[f'{prefix}amp_ratio_rib_abd'] = rib_std / abd_std
    else:
        features[f'{prefix}amp_ratio_rib_abd'] = np.nan

    return features

def extract_spo2_features(signal, fs, desat_threshold_percent=3.0, critical_threshold=90.0):
    """Extracts features from an SpO2 signal segment."""
    features = {}
    prefix = 'spo2_'

    if signal is None or len(signal) < int(fs * 2):
        # Return NaNs if signal is too short or None
        stats = _calculate_signal_stats(None)
        features.update({f"{prefix}{k}": v for k, v in stats.items()})
        features[f'{prefix}desat_count_{desat_threshold_percent}p'] = 0
        features[f'{prefix}desat_total_duration_{desat_threshold_percent}p'] = 0.0
        features[f'{prefix}desat_mean_nadir_{desat_threshold_percent}p'] = np.nan
        features[f'{prefix}time_below_{critical_threshold}'] = 0.0
        entropy = _calculate_entropy_features(None)
        features.update({f"{prefix}entropy_{k}": v for k, v in entropy.items()})
        features[f'{prefix}roc_mean'] = np.nan
        features[f'{prefix}roc_std'] = np.nan
        features[f'{prefix}roc_max_desat'] = np.nan
        features[f'{prefix}roc_max_resat'] = np.nan
        return features

    # Preprocessing (adjust median filter window for 8Hz)
    try:
        med_win_len = min(int(fs * 1.0) | 1, len(signal)) # 1 second median filter still okay
        if med_win_len >= 1:
             # Pad signal slightly before rolling median to reduce edge effects
             padded_signal = np.pad(signal, (med_win_len // 2, med_win_len // 2), mode='edge')
             signal_filt_series = pd.Series(padded_signal).rolling(window=med_win_len, center=True, min_periods=1).median()
             signal_filt = signal_filt_series[med_win_len // 2 : -med_win_len // 2].values # Extract original length
             # Fallback for potential all-NaN results after median filtering
             if np.all(np.isnan(signal_filt)): signal_filt = signal
             else: signal_filt = np.nan_to_num(signal_filt, nan=np.nanmean(signal)) # Fill remaining NaNs
        else:
             signal_filt = signal
    except Exception:
        signal_filt = signal

    stats = _calculate_signal_stats(signal_filt)
    features.update({f"{prefix}{k}": v for k, v in stats.items()})

    # Desaturation Event Detection
    num_desat_events = 0
    total_desat_duration_sec = 0
    desat_nadir_sum = 0
    time_below_critical = 0

    # Use a more robust baseline if possible (e.g., rolling median over longer window)
    # Simple version: median of the filtered segment
    baseline = np.median(signal_filt[~np.isnan(signal_filt)])
    if np.isnan(baseline): baseline = 98.0 # Default baseline if all NaN

    desat_level = baseline - desat_threshold_percent

    is_in_desat = False
    current_desat_start_idx = -1
    current_desat_nadir = 100.0

    for i in range(len(signal_filt)):
        if np.isnan(signal_filt[i]): continue # Skip NaN values

        if signal_filt[i] < critical_threshold:
            time_below_critical += 1

        if signal_filt[i] < desat_level and not is_in_desat:
            is_in_desat = True
            current_desat_start_idx = i
            current_desat_nadir = signal_filt[i]
        elif is_in_desat:
            current_desat_nadir = min(current_desat_nadir, signal_filt[i])
            # Event end criteria: signal must rise above desat level
            if signal_filt[i] >= desat_level:
                is_in_desat = False
                duration_samples = i - current_desat_start_idx
                min_duration_samples = int(1 * fs) # Require at least 1 second duration
                if duration_samples >= min_duration_samples:
                    num_desat_events += 1
                    total_desat_duration_sec += duration_samples / fs
                    desat_nadir_sum += current_desat_nadir

    if is_in_desat:
         duration_samples = len(signal_filt) - current_desat_start_idx
         min_duration_samples = int(1 * fs)
         if duration_samples >= min_duration_samples:
            num_desat_events += 1
            total_desat_duration_sec += duration_samples / fs
            desat_nadir_sum += current_desat_nadir

    features[f'{prefix}desat_count_{desat_threshold_percent}p'] = num_desat_events
    features[f'{prefix}desat_total_duration_{desat_threshold_percent}p'] = total_desat_duration_sec
    features[f'{prefix}desat_mean_nadir_{desat_threshold_percent}p'] = (desat_nadir_sum / num_desat_events) if num_desat_events > 0 else baseline
    features[f'{prefix}time_below_{critical_threshold}'] = time_below_critical / fs

    entropy = _calculate_entropy_features(signal_filt)
    features.update({f"{prefix}entropy_{k}": v for k, v in entropy.items()})

    rate_of_change = np.diff(signal_filt) * fs
    if len(rate_of_change[~np.isnan(rate_of_change)]) > 0: # Calculate only on non-NaN diffs
        valid_roc = rate_of_change[~np.isnan(rate_of_change)]
        features[f'{prefix}roc_mean'] = np.mean(valid_roc)
        features[f'{prefix}roc_std'] = np.std(valid_roc)
        features[f'{prefix}roc_max_desat'] = np.min(valid_roc)
        features[f'{prefix}roc_max_resat'] = np.max(valid_roc)
    else:
        features[f'{prefix}roc_mean'] = np.nan
        features[f'{prefix}roc_std'] = np.nan
        features[f'{prefix}roc_max_desat'] = np.nan
        features[f'{prefix}roc_max_resat'] = np.nan

    return features


# --- NEW Main Function for 3D Array Input ---

def extract_features_from_array(data, fs, channel_map):
    """
    Extracts features from a 3D NumPy array of physiological signals.

    Args:
        data (np.ndarray): Input data array with shape (n_samples, n_channels, seq_len).
                           `seq_len` should be 240 for 30s at 8Hz.
        fs (int): Sampling frequency (should be 8 Hz).
        channel_map (dict): Dictionary mapping channel index to signal name.
                            Example: {0: 'spo2', 1: 'nasal', 2: 'ribcage', 3: 'abdomen'}

    Returns:
        pd.DataFrame: DataFrame where each row corresponds to a sample (epoch)
                      and columns are the extracted features.
    """
    n_samples, n_channels, seq_len = data.shape
    print(f"Processing {n_samples} samples, {n_channels} channels, {seq_len} sequence length at {fs} Hz.")

    all_epochs_features_list = []

    # Create reverse map for easy lookup
    name_to_idx = {v: k for k, v in channel_map.items()}

    for i in range(n_samples):
        if (i + 1) % 100 == 0: # Print progress
            print(f"Processing sample {i+1}/{n_samples}...")

        epoch_features = {}
        current_epoch_slice = data[i, :, :] # Shape: (n_channels, seq_len)

        # Get individual signals based on channel_map
        spo2_signal = current_epoch_slice[name_to_idx['spo2'], :] if 'spo2' in name_to_idx else None
        nasal_signal = current_epoch_slice[name_to_idx['nasal'], :] if 'nasal' in name_to_idx else None
        rib_signal = current_epoch_slice[name_to_idx['ribcage'], :] if 'ribcage' in name_to_idx else None
        abd_signal = current_epoch_slice[name_to_idx['abdomen'], :] if 'abdomen' in name_to_idx else None

        # --- Call individual feature extractors ---
        if spo2_signal is not None:
            spo2_features = extract_spo2_features(spo2_signal, fs)
            epoch_features.update(spo2_features)

        if nasal_signal is not None:
            # Tune parameters specifically for nasal signal if needed
            nasal_features = extract_respiratory_features(nasal_signal, fs, prefix='nasal_', peak_prominence_ratio=0.1, peak_distance_sec=1.0)
            epoch_features.update(nasal_features)

        if rib_signal is not None:
            rib_features = extract_respiratory_features(rib_signal, fs, prefix='rib_', peak_prominence_ratio=0.1, peak_distance_sec=1.0)
            epoch_features.update(rib_features)

        if abd_signal is not None:
            abd_features = extract_respiratory_features(abd_signal, fs, prefix='abd_', peak_prominence_ratio=0.1, peak_distance_sec=1.0)
            epoch_features.update(abd_features)

        # Extract Thoracoabdominal Synchronization features
        if rib_signal is not None and abd_signal is not None:
            sync_features = extract_thoracoabdominal_features(rib_signal, abd_signal, fs)
            epoch_features.update(sync_features)

        all_epochs_features_list.append(epoch_features)

    print("Feature extraction complete.")
    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(all_epochs_features_list)
    return features_df


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    FS = 8  # Sampling frequency in Hz
    EPOCH_DURATION_SEC = 30
    SEQ_LEN = FS * EPOCH_DURATION_SEC # Should be 240

    N_SAMPLES = 5 # Number of example epochs
    N_CHANNELS = 4 # spo2, nasal, ribcage, abdomen

    # IMPORTANT: Define your channel mapping based on your data file
    CHANNEL_MAP = {
        0: 'spo2',
        1: 'nasal',
        2: 'ribcage',
        3: 'abdomen'
    }
    # Verify map matches N_CHANNELS
    assert len(CHANNEL_MAP) == N_CHANNELS, "Channel map size mismatch"



    # --- Extract Features ---
    loaded_data = np.load('BetterModels/sleep_dataset.npz', allow_pickle=True)
    X = [loaded_data[f'X_{i}'] for i in range(25)]
    Y = [loaded_data[f'Y_{i}'] for i in range(25)]
    # X = np.vstack(X)
    # features_dataframe = extract_features_from_array(X, FS, CHANNEL_MAP)
    X_featurized = []
    for i in range(len(X)):
        X_featurized.append(extract_features_from_array(X[i], FS, CHANNEL_MAP))
        print(f"Loaded data shape: {X[i].shape}")
        print(f"Featurized data shape: {X_featurized[0].shape}")

    print(f"Featurized data shape: {X_featurized[0].shape}")
    print(len(X_featurized))
    # --- Save Features ---
    np.savez('sleep_dataset_features.npz', 
            **{f'X_{i}': x for i, x in enumerate(X_featurized)},
            **{f'Y_{i}': y for i, y in enumerate(Y)})
    # --- Display Results ---
    # print("\nFeature DataFrame:")
    # pd.set_option('display.max_rows', 10) # Show first few rows
    # pd.set_option('display.max_columns', None) # Show all columns
    # print(features_dataframe)

    # # Check for NaN values (can indicate issues with extraction/data)
    # print(f"\nTotal NaN values in DataFrame: {features_dataframe.isnull().sum().sum()}")
    # print("\nNaN counts per feature (first 20):")
    # print(features_dataframe.isnull().sum().head(20))