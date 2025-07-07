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


# ---------------------------------------------------------------------------------------------------------------------------------

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



# ---------------------------------------------------------------------------------------------------------------------------------


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