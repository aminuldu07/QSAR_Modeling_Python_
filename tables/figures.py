"""

Module script containing caching for the table
data in the manuscript.

"""
from send import send_db
from hepatotox_db import ANIMALDB
from config import IMG_DIR
import os
import pandas as pd

from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt

figures_dir = IMG_DIR

data = pd.read_csv('../data_scratch/new_rat_data.csv', index_col=0)
steatosis = set(data.USUBJID[data['STEATOSIS'] == 1])
cholestasis = set(data.USUBJID[data['CHOLESTASIS'] == 1])
necrosis = set(data.USUBJID[data['NECROSIS'] == 1])

print(len(steatosis))

plt.figure(figsize=(8, 8))

v = venn3(subsets=[steatosis, cholestasis, necrosis],
          set_labels=('Steatosis', 'Cholestasis', 'Necrosis'))

venn3_circles(subsets=[steatosis, cholestasis, necrosis])

plt.savefig(os.path.join(figures_dir, 'venn.png'), transparent=True)

# Figure 2.
# Histogram of lab tests

tests = data[[col for col in data.columns if '-' in col]].notnull().sum().sort_values(ascending=False)

print(tests.shape)

tests = tests[tests > 2000]

print(tests.shape)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

ax.bar(tests.index, tests.values, facecolor=(0, 0.5, 0.8, 0.4), edgecolor='k', linewidth=1.5)

ax.set_xticklabels(list(tests.index), rotation='vertical')

ax.set_ylabel("Animals tested", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'Figure2.png'), transparent=True)

