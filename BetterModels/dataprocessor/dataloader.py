import pyedflib
import numpy as np

def read_edf_signals(edf_file):
    f = pyedflib.EdfReader(edf_file)
    num_signals = f.signals_in_file
    signals = {}
    for i in range(num_signals):
        signal = f.readSignal(i)
        label = f.getSignalLabels()[i]
        signals[label] = signal

    labels = f.getSignalLabels()
    sampling_rates = f.getSampleFrequencies()

    f.close()
    
    return signals, labels, sampling_rates

edf_file_path = "./dataset/files/ucddb002.rec"
signals, labels, sampling_rates = read_edf_signals(edf_file_path)

print("Signal labels:", labels)
print("Sampling rates:", sampling_rates)
print("First 10 samples of the first signal:", signals["Lefteye"][:10]) 

# Imp signals -> 'Flow', 'ribcage', 'abdomen', 'SpO2', 'sum'
# 6 -> SpO2   
# 8 -> Flow
# 9 -> Sum
# 10 -> ribcage
# 11 -> abdomen


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

        # shape -> (size ,dimensions) -> (n_timesteps, n_channels)
        print("Shape of signals:", signals['SpO2'].shape)


        # signal_keys = ['SpO2', 'Flow', 'Sum', 'ribcage', 'abdo']
        signal_keys = ['C3A2', 'C4A1']
        signals_stacked = np.column_stack([signals[key] for key in signal_keys])

        print("Shape of signal stacked:", signals_stacked.shape)
        print("Shape of sleep stages:", sleep_stages.shape)

        print("signals range:", signals_stacked[:,0].min(), signals_stacked[:,0].max())
        mu = signals_stacked.mean(axis=0)      # per-channel mean
        sigma = signals_stacked.std(axis=0)       # per-channel std
        sigma[sigma == 0] = 1e-8
        normalized_signals_stacked = (signals_stacked - mu) / sigma
        print("Normalized signals shape:", normalized_signals_stacked.shape)
        print("normalized signals range:", normalized_signals_stacked[:,0].min(), normalized_signals_stacked[:,0].max())

        for key in signal_keys:
            signals[key] = signals[key][:len(sleep_stages)*3840]

        sleep_stages_remove_indices = []
        signal_remove_indices = []
        for j in range(len(sleep_stages)):
            if sleep_stages[j] == 8:
                sleep_stages_remove_indices.append(j)
                _indices = np.arange(j*3840, (j+1)*3840)
                signal_remove_indices.extend(_indices)

        for key in signal_keys:
            signals[key] = np.delete(signals[key], signal_remove_indices)
            signals[key] = signals[key].reshape(-1, 3840)
        sleep_stages = np.delete(sleep_stages, sleep_stages_remove_indices)
        sleep_stages = sleep_stages.reshape(-1, 1)

        signals_stacked = np.stack([signals[key] for key in signal_keys], axis=1)
        print("Shape of signals:", signals_stacked.shape)
        print("Shape of sleep stages:", sleep_stages.shape)
        x_list.append(signals_stacked)
        y_list.append(sleep_stages)
    except:
        print(f"Error in file {i}")
    # continue

print("-------------------------------------------------------------------------------------------")
for i in range(len(x_list)):
     print(f"File {i} has {len(x_list)} signals", \
        "\nsamples [nasal, ribcage]", f"[{len(signals['C4A1'])}, {len(signals['C3A2'])}]", \
        "\nsampling rate [nasal, ribcage]", f"[{sampling_rates[3]}, {sampling_rates[4]}]",\
        "\nno of sleep stages", len(x_list[i])* len(x_list[0][1][0])/(30*sampling_rates[3]), len(x_list[i])* len(x_list[0][1][0])/(30*sampling_rates[4]),  \
        "\nSleep stages: ", len(y_list[i])) 
     print()



np.savez('sleep_dataset_essential.npz', 
         **{f'X_{i}': x for i, x in enumerate(x_list)},
         **{f'Y_{i}': y for i, y in enumerate(y_list)})





# import collections 

# Car = collections.namedtuple('Car', ['make', 'model', 'year'])
# a = Car(make='Toyota', model='Corolla', year=2020)
# print(a.make, a.model, a.year)




# Card = collections.namedtuple('Card', ['rank', 'suit'])

# class FrenchDeck():
#     ranks = [str(n) for n in range(2, 11)] + list('JQKA')
#     suits = ['hearts', 'diamonds', 'clubs', 'spades']

#     def __init__(self):
#         self._cards = [Card(rank, suit) for rank in self.ranks for suit in self.suits]
    
#     def __len__(self):
#         return len(self._cards)
    
#     def __getitem__(self, idx):
#         return self._cards[idx]
    

# beer_card = Card('7', 'diamonds')
# print(beer_card.rank, beer_card.suit)


# suit_values = dict(hearts=2, diamonds=1, clubs=0, spades=3)
# deck = FrenchDeck()

# def spades_high(card):
#     """Return a value for the card."""
#     rank_values = FrenchDeck.ranks.index(card.rank)
#     return suit_values[card.suit] + rank_values*4

# for card in sorted(deck, key=spades_high): # doctest: +ELLIPSIS
#     print(card)
