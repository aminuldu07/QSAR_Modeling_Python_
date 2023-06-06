import matplotlib.pyplot as plt
from clincal_chemistry import get_liver_clin_chem, all_tests
import config, os, numpy as np, pandas as pd
from send import send_db

dm = send_db.generic_query('SELECT STUDYID, USUBJID, ARMCD, SETCD, SPECIES, SEX FROM DM')

animals = send_db.generic_query('SELECT STUDYID, TSVAL FROM TS WHERE TSPARMCD = "SPECIES"')
animals.columns = ['STUDYID', 'TSVAL']





dm = dm[dm.SEX.notnull()]
data = dm.merge(animals)

# clean up species
data['SPECIES'] = data['SPECIES'].where(data['SPECIES'] == ' ', data['TSVAL'])
data['SPECIES'] = [spec.upper() if spec != 'Rats' else 'RAT' for spec in data['SPECIES']]
data.loc[data.SPECIES == 'MICE', 'SPECIES'] = 'MOUSE'

matrix = get_liver_clin_chem()

matrix = matrix.reset_index().merge(data[['USUBJID', 'SPECIES']])

matrix.set_index('USUBJID', inplace=True)



ordered_cols = matrix[all_tests].notnull().sum().sort_values(ascending=False).index

not_nulls = matrix[ordered_cols].notnull().sum().values
is_nulls = matrix[ordered_cols].isnull().sum().values

idx = np.arange(0, len(ordered_cols))

fig, ax = plt.subplots(figsize=(10, 10))

bottom = np.zeros(len(not_nulls))

for species in matrix.SPECIES.value_counts().index:

    species_matrix = matrix[matrix.SPECIES == species]
    not_nulls = species_matrix[ordered_cols].notnull().sum().values

    ax.bar(idx, not_nulls, bottom=bottom, edgecolor='black', lw=2, label=species)
    bottom = bottom+not_nulls

ax.bar(idx, is_nulls, bottom=bottom, color=(0, 0, 0, 0.4), hatch='//', edgecolor='black', lw=2, label='Null')

ax.set_xticks(list(range(0, len(ordered_cols))))
ax.set_xticklabels(ordered_cols, fontsize=14)

ax.legend()

ax.set_title("Frequency of Clincal Chemistry Lab Values Across {} Animals".format(matrix.shape[0]))

plt_filename = os.path.join(config.IMG_DIR, 'lb_clin_test.png')

plt.savefig(plt_filename, transparent=True)
