import pandas as pd
import config, os
import matplotlib.pyplot as plt

stats = pd.read_csv(os.path.join(config.TEXT_DIR, 'CROSS_VALIDATION_RESULTS.csv'), index_col=[0, 1], header=[0, 1])

avg_stats = stats['AVG_VALUES'].reset_index()
std_stats = stats['STD_VALUES'].reset_index()

valid_stats = ['ACC', 'Precision', 'Recall', 'Specificity']

other_cols = ['DISEASE', 'N_SPLITS']

avg_stats = avg_stats[other_cols + valid_stats]
std_stats = std_stats[other_cols + valid_stats]

fig, axarr = plt.subplots(3, 1)

idx = 0

for disease, avg_data in avg_stats.groupby('DISEASE'):
    std_data = std_stats[std_stats.DISEASE == disease]

    for stat in valid_stats:
        axarr[idx].scatter([0, 1, 2, 3], avg_data[stat])

    idx = idx + 1

for ax in axarr:
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([5, 10, 20, 100])



plt.show()