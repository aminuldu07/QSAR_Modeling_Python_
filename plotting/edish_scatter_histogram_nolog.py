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


def color_findings(findings):
    if findings['NECROSIS']:
        return 1, 0, 0, 1
    if findings['STEATOSIS']:
        return 1, 1, 0, 1
    if findings['CHOLESTASIS']:
        return 1, 165/255, 0, 1
    return 0.8, 0.8, 0.8, 1

animals = get_experimental_animals('RAT')
findings = get_classified_liver_results()
data = get_clean_liver_clin_chem()

data = pd.merge(data.reset_index(), animals[['USUBJID', 'SEX', 'MAXDOSE', 'CNTRL_GRP']])
data = pd.merge(data, findings)

data.set_index('USUBJID', inplace=True)

data = data[data[['ALT', 'BILI']].notnull().all(1)]

data['ALT_NORM'] = None
data['BILI_NORM'] = None

# plotting parameters

sex = 'M'

sex_colors = {'M': (176/255, 224/255, 230/255, 0.6),
              'F': (255/255, 192/255, 203/255, 1)}


sex_color = sex_colors[sex]

data = data[data.SEX == sex]

for test in ['ALT', 'BILI']:
    data.loc[data[test] < 0, test] = 0


data['ALT_NORM'] = data.ALT / config.lab_test_parameters['ALT'][sex]
data['BILI_NORM'] = data.BILI / config.lab_test_parameters['BILI'][sex]

scale_shift = 1

data.BILI_NORM = data.BILI_NORM + scale_shift
data.ALT_NORM = data.ALT_NORM + scale_shift

data.BILI_NORM = data.BILI_NORM.apply(lambda x: math.log10(x))
data.ALT_NORM = data.ALT_NORM.apply(lambda x: math.log10(x))

# data.BILI_NORM = data.BILI_NORM - scale_shift
# data.ALT_NORM = data.ALT_NORM - scale_shift

three_times_alt_norm = math.log10((config.lab_test_parameters['ALT'][sex] * 3 /config.lab_test_parameters['ALT'][sex]))
two_times_bili_norm = math.log10((config.lab_test_parameters['BILI'][sex] * 2 /config.lab_test_parameters['BILI'][sex]))


nullfmt = NullFormatter()         # no labels

# definitions for the axes



#
#
#
#
#
alpha = 0.4
s = 125


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
xymax = np.max([np.max(np.fabs(data.ALT_NORM.values)), np.max(np.fabs(data.BILI_NORM.values))])
lim = (int(xymax / binwidth) + 1) * binwidth

# axScatter.set_xlim((-lim, lim))
# axScatter.set_ylim((-lim, lim))

# bins = np.arange(-lim, lim + binwidth, binwidth)


x_heights, x_mids, _ = axHistx.hist(data.ALT_NORM[data.MAXDOSE == 0 & data.CNTRL_GRP], bins=50, facecolor=sex_color, edgecolor='k')
y_heights, y_mids, _  = axHisty.hist(data.BILI_NORM[data.MAXDOSE == 0 & data.CNTRL_GRP], bins=50, facecolor=sex_color, edgecolor='k', orientation='horizontal')

axScatter.scatter(data.ALT_NORM[data.MAXDOSE == 0 & data.CNTRL_GRP],
                  data.BILI_NORM[data.MAXDOSE == 0 & data.CNTRL_GRP], s=40,
                  facecolor=sex_color, edgecolor='k', zorder=1)


axHistx.plot([three_times_alt_norm, three_times_alt_norm], [0, x_heights.max()], 'k--', zorder=0)
axScatter.plot([three_times_alt_norm, three_times_alt_norm], [0, data.BILI_NORM.max()], 'k--', zorder=0)

axHisty.plot([0, y_heights.max()], [two_times_bili_norm, two_times_bili_norm], 'k--', zorder=0)
axScatter.plot([0, data.ALT_NORM.max()], [two_times_bili_norm, two_times_bili_norm], 'k--', zorder=0)

axScatter.scatter(data.ALT_NORM[data.MAXDOSE != 0],
                  data.BILI_NORM[data.MAXDOSE != 0], s=75,
                  facecolor=[color_findings(find) for i, find in data.loc[data.MAXDOSE != 0, ['STEATOSIS', 'CHOLESTASIS', 'NECROSIS']].iterrows()],
                  edgecolor='k', zorder=0)


axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

axHistx.tick_params(labelsize=18)
axHisty.tick_params(labelsize=18)
axScatter.tick_params(labelsize=18)




# handles, labels = axScatter.get_legend_handles_labels()
#
# axScatter.legend(handles[::-1], labels[::-1], fontsize=22, loc='best')

axScatter.set_xlabel('log10((ALT + 1 / ULT)', fontsize=22)
axScatter.set_ylabel('log10((BILI + 1 / ULT)', fontsize=22)


axScatter.scatter([], [], s=75,
                  facecolor=sex_color,
                  edgecolor='k', zorder=0, label='Control')
axScatter.scatter([], [], s=75,
                  facecolor=(0.8, 0.8, 0.8, 1),
                  edgecolor='k', zorder=0, label='Normal')
axScatter.scatter([], [], s=75,
                  facecolor=(1, 0, 0, 1),
                  edgecolor='k', zorder=0, label='Necrosis')
axScatter.scatter([], [], s=75,
                  facecolor=(1, 1, 0, 1),
                  edgecolor='k', zorder=0, label='Steatosis')
axScatter.scatter([], [], s=75,
                  facecolor=(1, 165 / 255, 0, 1),
                  edgecolor='k', zorder=0, label='Cholestasis')
axScatter.legend(fontsize=18)

# axScatter.set_yticks([axScatter.get_yticks().min(), axScatter.get_yticks().max()])
# axScatter.set_xticks([axScatter.get_xticks().min(), axScatter.get_xticks().max()])
plt.savefig(os.path.join(config.IMG_DIR, 'edish_{}_nolog.png'.format(sex)), transparent=True)