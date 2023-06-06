import pandas as pd
import glob
import matplotlib.pylab as plt
from matplotlib.ticker import NullFormatter
from numpy.polynomial.polynomial import polyfit
import config

import os, math
import numpy as np

from send import send_db

df = pd.read_csv('../data_scratch/new_rat_data.csv', index_col=0)
data = df.replace(np.inf, np.nan)
data = data[data[['ALT-SERUM', 'BILI-SERUM']].notnull().all(1)]

data[['ALT-SERUM', 'BILI-SERUM']] = data[['ALT-SERUM', 'BILI-SERUM']].applymap(lambda x: math.log10(x+1))

sex = 'M'

sex_colors = {'M': (176/255, 224/255, 230/255, 0.6),
              'F': (255/255, 192/255, 203/255, 1)}


sex_color = sex_colors[sex]



alt_lim = math.log10(3+1)
bili_lim = math.log10(2+1)

nullfmt = NullFormatter()         # no labels

data = data[(data['ALT-SERUM'] > alt_lim) & (data['BILI-SERUM'])]

mistresc = send_db.generic_query('SELECT USUBJID, upper(MISTRESC) as MISTRESC FROM MI WHERE MISPEC == "LIVER"')
mistresc = mistresc[mistresc.USUBJID.isin(data.USUBJID)]
mistresc.to_csv('../data_scratch/liver_findings_above_threshold.csv')
# mistresc['liver_results'] = mistresc.groupby('USUBJID')['MISTRESC'].transform(lambda x: '; '.join(x))