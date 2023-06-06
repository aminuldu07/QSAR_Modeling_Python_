import os, config
import pandas as pd

data = pd.read_csv(os.path.join(config.TEXT_DIR, 'mi_liver_signals.csv'))

inds_results = data.groupby(['APPNUMBER', 'MISTRESC'])

for ind_result_pair, pair_data in inds_results:
    print(pair_data)
