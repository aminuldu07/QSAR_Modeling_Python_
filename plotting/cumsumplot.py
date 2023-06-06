import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os, config
from stats import get_class_stats

diseases = ['NECROSIS', 'CHOLESTASIS', 'STEATOSIS']
txt_dir = os.path.join(config.TEXT_DIR, 'pls_results', 'BOTH_DILI')
data_to_cache = os.path.join(txt_dir, 'best_predictions_BOTH_DILI.csv')

data = pd.read_csv(data_to_cache, index_col=0)
# data.MISTRESC = data.MISTRESC.str.upper().fillna('')
# data.sort_values('PREDICTION_MEAN', ascending=False, inplace=True)
# # data = data[data.PREDICTION_MEAN > 0.75]
# data.IS_CONTROL = ((data.MISTRESC == 'NORMAL') | (data.MISTRESC == 'UNREMARKABLE')).astype(int)
# data.PER_NORMAL = data.IS_CONTROL.cumsum() / data.IS_CONTROL.sum()
# data['RANK'] = np.arange(data.shape[0])

# offset = 0.2
# final_data = data[(data.PREDICTION_MEAN < 0.5 - offset) | (data.PREDICTION_MEAN >= 0.5 + offset)]
#
#
#
# stats = get_class_stats(None, final_data['DIRECT_DILI'], final_data.PREDICTION_MEAN_CLASS)
#
# print(stats)