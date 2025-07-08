# create binary, 3, 4, 6 stage datasets
import numpy as np
loaded = np.load("BetterModels/datastore/sleep_dataset_essential.npz", allow_pickle=True)
# loaded = np.load("BetterModels/sleep_dataset_features.npz", allow_pickle=True)

X_list_loaded = [loaded[f'X_{i}'] for i in range(25)]
Y_list_loaded = [loaded[f'Y_{i}'] for i in range(25)]

print("-------------------------------------------------------------------------------------------")
print("Shape of signals:", X_list_loaded[0].shape)
print("Shape of sleep stages:", Y_list_loaded[0].shape)
print("-------------------------------------------------------------------------------------------")

print(Y_list_loaded[0].shape)
# print(np.unique(Y_list_loaded, return_counts=True))

y_stage_list = []
for i in range(len(Y_list_loaded)):
    # 0 -> wake, 1 -> REM, 2,3 -> Light Sleep, 4,5 -> Deep Sleep
    y_stage = np.zeros_like(Y_list_loaded[i])
    y_stage[Y_list_loaded[i] == 1] = 1 # REM
    y_stage[Y_list_loaded[i] == 2] = 2
    y_stage[Y_list_loaded[i] == 3] = 3
    y_stage[Y_list_loaded[i] == 4] = 4
    y_stage[Y_list_loaded[i] == 5] = 5
    

    # 0 -> wake, 1 -> REM, 2 -> Light Sleep, 3 -> Deep Sleep

    y_stage_list.append(y_stage)
print(np.unique(y_stage_list[0], return_counts=True))

y_binary_list = []
for i in range(len(Y_list_loaded)):
    y_binary = (Y_list_loaded[i] > 0).astype(int)
    y_binary_list.append(y_binary)
print(np.unique(y_binary_list[0], return_counts=True))

print("-------------------------------------------------------------------------------------------")
print("Shape of signals:", X_list_loaded[0].shape)
print("Shape of sleep stages:", Y_list_loaded[0].shape)
print(len(X_list_loaded), len(Y_list_loaded))
np.savez('BetterModels/datastore/2_stage_sleep_dataset_essential.npz', 
         **{f'X_{i}': x for i, x in enumerate(X_list_loaded)},
         **{f'Y_{i}': y for i, y in enumerate(y_binary_list)})
# np.savez('2_stage_sleep_dataset_features.npz', 
#          **{f'X_{i}': x for i, x in enumerate(X_list_loaded)},
#          **{f'Y_{i}': y for i, y in enumerate(y_binary_list)})


y_3_stage_list = []
for i in range(len(Y_list_loaded)):
    # 0 -> wake, 1 -> REM, 2,3,4,5 -> NREM
    y_3_stage = np.zeros_like(Y_list_loaded[i])
    y_3_stage[Y_list_loaded[i] == 1] = 1 # REM
    y_3_stage[Y_list_loaded[i] == 2] = 2
    y_3_stage[Y_list_loaded[i] == 3] = 2
    y_3_stage[Y_list_loaded[i] == 4] = 2
    y_3_stage[Y_list_loaded[i] == 5] = 2

    # 0 -> wake, 1 -> REM, 2 -> NREM
    y_3_stage_list.append(y_3_stage)

print(np.unique(y_3_stage_list[0], return_counts=True))
np.savez('BetterModels/datastore/3_stage_sleep_dataset_essential.npz', 
         **{f'X_{i}': x for i, x in enumerate(X_list_loaded)},
         **{f'Y_{i}': y for i, y in enumerate(y_3_stage_list)})
# np.savez('3_stage_sleep_features.npz', 
#          **{f'X_{i}': x for i, x in enumerate(X_list_loaded)},
#          **{f'Y_{i}': y for i, y in enumerate(y_3_stage_list)})


y_4_stage_list = []
for i in range(len(Y_list_loaded)):
    # 0 -> wake, 1 -> REM, 2,3 -> Light Sleep, 4,5 -> Deep Sleep
    y_4_stage = np.zeros_like(Y_list_loaded[i])
    y_4_stage[Y_list_loaded[i] == 1] = 1 # REM
    y_4_stage[Y_list_loaded[i] == 2] = 2
    y_4_stage[Y_list_loaded[i] == 3] = 2
    y_4_stage[Y_list_loaded[i] == 4] = 3
    y_4_stage[Y_list_loaded[i] == 5] = 3

    # 0 -> wake, 1 -> REM, 2 -> Light Sleep, 3 -> Deep Sleep

    y_4_stage_list.append(y_4_stage)
print(np.unique(y_4_stage_list[0], return_counts=True))
np.savez('BetterModels/datastore/4_stage_sleep_dataset_essential.npz', 
         **{f'X_{i}': x for i, x in enumerate(X_list_loaded)},
         **{f'Y_{i}': y for i, y in enumerate(y_4_stage_list)})
# np.savez('4_stage_sleep_features.npz', 
#          **{f'X_{i}': x for i, x in enumerate(X_list_loaded)},
#          **{f'Y_{i}': y for i, y in enumerate(y_4_stage_list)})


print(np.unique(y_3_stage_list[0], return_counts=True))
# print(np.unique(y_binary_list[0], return_counts=True))
print(np.unique(y_4_stage_list[0], return_counts=True))
np.unique(Y_list_loaded[0], return_counts=True)

