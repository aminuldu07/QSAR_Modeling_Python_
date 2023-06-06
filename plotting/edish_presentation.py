import pandas as pd
import glob
import matplotlib.pylab as plt
from matplotlib.ticker import NullFormatter
from numpy.polynomial.polynomial import polyfit
import config

import os, math
import numpy as np

from clincal_chemistry import get_clean_liver_clin_chem
from morphological_findings import get_classified_liver_results
from experimental_animals import get_experimental_animals

df = pd.read_csv('../data_scratch/new_rat_data.csv', index_col=0)
data = df.replace(np.inf, np.nan)
data = data[data[['ALT-SERUM', 'BILI-SERUM']].notnull().all(1)]

data[['ALT-SERUM', 'BILI-SERUM']] = data[['ALT-SERUM', 'BILI-SERUM']].applymap(lambda x: math.log10(x+1))

sex = 'M'

sex_colors = {'M': (176/255, 224/255, 230/255, 0.6),
              'F': (255/255, 192/255, 203/255, 1)}


sex_color = sex_colors[sex]

data = data[data.SEX == sex]

alt_lim = math.log10(3+1)
bili_lim = math.log10(2+1)

nullfmt = NullFormatter()         # no labels

# definitions for the axes



#
#
#
#
#
alpha = 0.4
s = 125
# colors = [(1, 0, 0, alpha), (0, 1, 0, alpha), (0, 0, 1, alpha), (0, 1, 1, alpha)]

left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02
bottom_h_consensus = left_h + width + bottom_h + 0.2

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(10, 10))

axScatter = plt.axes(rect_scatter, label='scatter')
axHistx = plt.axes(rect_histx, label='xhist')
axHisty = plt.axes(rect_histy,  label='yhist')


# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)


# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max([np.max(np.fabs(data['ALT-SERUM'].values)), np.max(np.fabs(data['BILI-SERUM']))])
lim = (int(xymax / binwidth) + 1) * binwidth

# axScatter.set_xlim((-lim, lim))
# axScatter.set_ylim((-lim, lim))

# bins = np.arange(-lim, lim + binwidth, binwidth)


x_heights, x_mids, _ = axHistx.hist(data['ALT-SERUM'], bins=50, facecolor=sex_color, edgecolor='k')
y_heights, y_mids, _  = axHisty.hist(data['BILI-SERUM'], bins=50, facecolor=sex_color, edgecolor='k', orientation='horizontal')


colored = True

if colored:
    axScatter.scatter(data['ALT-SERUM'][data.NECROSIS == 0],
                      data['BILI-SERUM'][data.NECROSIS == 0], s=60,
                      facecolor=sex_color, edgecolor='k', zorder=1)
    axScatter.scatter(data['ALT-SERUM'][data.NECROSIS == 1],
                      data['BILI-SERUM'][data.NECROSIS == 1], s=60,
                      facecolor=(1, 0, 0, 1), edgecolor='k', zorder=1)
else:
    axScatter.scatter(data['ALT-SERUM'],
                  data['BILI-SERUM'], s=60,
                  facecolor=sex_color, edgecolor='k', zorder=1)


axHistx.plot([alt_lim, alt_lim], [0, x_heights.max()], 'k--', zorder=0)
axScatter.plot([alt_lim, alt_lim], [0, data['BILI-SERUM'].max()], 'k--', zorder=0)

axHisty.plot([0, y_heights.max()], [bili_lim, bili_lim], 'k--', zorder=0)
axScatter.plot([0, data['ALT-SERUM'].max()], [bili_lim, bili_lim], 'k--', zorder=0)




axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

axHistx.tick_params(labelsize=18)
axHisty.tick_params(labelsize=18)
axScatter.tick_params(labelsize=18)

axScatter.set_xticks([])
axScatter.set_yticks([])


# handles, labels = axScatter.get_legend_handles_labels()
#
# axScatter.legend(handles[::-1], labels[::-1], fontsize=22, loc='best')

axScatter.set_xlabel('ALT', fontsize=22)
axScatter.set_ylabel('BILI', fontsize=22)


# axScatter.scatter([], [], s=75,
#                   facecolor=sex_color,
#                   edgecolor='k', zorder=0, label='Control')
# axScatter.scatter([], [], s=75,
#                   facecolor=(0.8, 0.8, 0.8, 1),
#                   edgecolor='k', zorder=0, label='Normal')
# axScatter.scatter([], [], s=75,
#                   facecolor=(1, 0, 0, 1),
#                   edgecolor='k', zorder=0, label='Necrosis')
# axScatter.scatter([], [], s=75,
#                   facecolor=(1, 1, 0, 1),
#                   edgecolor='k', zorder=0, label='Steatosis')
# axScatter.scatter([], [], s=75,
#                   facecolor=(1, 165 / 255, 0, 1),
#                   edgecolor='k', zorder=0, label='Cholestasis')
# axScatter.legend(fontsize=18)

# axScatter.set_yticks([axScatter.get_yticks().min(), axScatter.get_yticks().max()])
# axScatter.set_xticks([axScatter.get_xticks().min(), axScatter.get_xticks().max()])
plt.savefig(os.path.join(config.IMG_DIR, 'edish_{}_FINAL.png'.format(sex)), transparent=True)