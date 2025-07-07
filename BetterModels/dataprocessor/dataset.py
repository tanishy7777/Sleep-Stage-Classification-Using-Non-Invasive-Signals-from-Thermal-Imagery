
class SleepSequenceDataset(Dataset):
    """
    This dataset creates overlapping sequences of X.
    Each sample is a sequence of consecutive X (e.g., previous, current, next),
    with the middle epoch’s label used as the target.
    
    Parameters:
        X: a tensor of shape (n_samples, n_channels, n_timesteps)
        Y: a tensor of shape (n_samples, 1)  (or (n_samples,) works as well)
        seq_length: number of timeseries to include to predict current per sequence
    """
    def __init__(self, X, Y, seq_length=3):
        self.X = X
        self.Y = Y
        self.seq_length = seq_length
        self.half_seq = seq_length // 2
        # We can only form sequences when we have enough X on each side.
        self.num_sequences = len(X) - 2 * self.half_seq

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Get a sequence of X starting at idx, ending at idx+seq_length
        sequence = self.X[idx : idx + self.seq_length]  # shape: (seq_length, n_channels, n_timesteps)
        # Use the middle epoch’s label as the target
        target = self.Y[idx + self.half_seq]
        return sequence, target
