import pandas as pd
import glob
import matplotlib.pylab as plt
from matplotlib.ticker import NullFormatter
from numpy.polynomial.polynomial import polyfit
import config
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

import os, math
import numpy as np

from clincal_chemistry import get_clean_liver_clin_chem
from morphological_findings import get_classified_liver_results
from experimental_animals import get_experimental_animals

df = pd.read_csv('../data_scratch/new_rat_data.csv', index_col=0)
data = df.replace(np.inf, np.nan)
data = data[data[['ALT-SERUM', 'BILI-SERUM']].notnull().all(1)]

data[['ALT-SERUM', 'BILI-SERUM']] = data[['ALT-SERUM', 'BILI-SERUM']].applymap(lambda x: math.log10(x+1))

sex = 'F'

sex_colors = {'M': (176/255, 224/255, 230/255, 0.6),
              'F': (255/255, 192/255, 203/255, 1)}
all_data = data.copy()

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

# axScatter.scatter(data['ALT-SERUM'][~((data['ALT-SERUM'] > alt_lim) & (data['BILI-SERUM'] > bili_lim))],
#                   data['BILI-SERUM'][~((data['ALT-SERUM'] > alt_lim) & (data['BILI-SERUM'] > bili_lim))], s=40,
#                   facecolor=sex_color, edgecolor='k', zorder=1)
#
# axScatter.scatter(data['ALT-SERUM'][(data['ALT-SERUM'] > alt_lim) & (data['BILI-SERUM'] > bili_lim)],
#                   data['BILI-SERUM'][(data['ALT-SERUM'] > alt_lim) & (data['BILI-SERUM'] > bili_lim)], s=60,
#                   facecolor=sex_color, edgecolor='k', zorder=1)

axScatter.scatter(data['ALT-SERUM'][data.NECROSIS == 0],
                  data['BILI-SERUM'][data.NECROSIS == 0], s=60,
                  facecolor=sex_color, edgecolor='k', zorder=1)
axScatter.scatter(data['ALT-SERUM'][data.NECROSIS == 1],
                  data['BILI-SERUM'][data.NECROSIS == 1], s=60,
                  facecolor=(1, 0, 0, 1), edgecolor='k', zorder=1)


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
# plt.savefig(os.path.join(config.IMG_DIR, 'edish_{}_FINAL.png'.format(sex)), transparent=True)

le = LabelEncoder()


all_data['SEX'] = le.fit_transform(all_data['SEX'])
data['SEX'] = le.transform(data['SEX'])

X = all_data[['ALT-SERUM', 'BILI-SERUM', 'SEX']]
X.index = all_data.USUBJID
y = all_data['NECROSIS']
y.index = all_data.USUBJID

actives = X[y == 1]
inactives = X[y == 0].sample(n=int(actives.shape[0]*3))

X = pd.concat([actives, inactives])
y = y[X.index]

mdl = SVC(kernel='poly', class_weight='balanced',  C=25, probability=True)

mdl.fit(X, y)

cv = CalibratedClassifierCV(mdl, cv='prefit', method='sigmoid')

x_lims = axScatter.get_xlim()
y_lims = axScatter.get_ylim()

xx, yy = np.mgrid[x_lims[0]:x_lims[1]:0.01, y_lims[0]:y_lims[1]:0.01]

#
fake_data = np.c_[xx.ravel(),
                  yy.ravel(),
                  np.full(xx.ravel().shape, data.SEX.mean())]
#
probs = mdl.predict_proba(fake_data)[:, 1].reshape(xx.shape)

output_file = os.path.join(os.getenv('LIVER_TOX'), 'img', 'good_figures', 'contour.png')

# xx = xx-np.log10(2000)
# yy = yy-np.log10(2000)

# xx = alt_convert(xx)
# yy = bili_convert(yy)

contour = axScatter.contourf(xx, yy, probs, 25, cmap="RdBu_r", alpha=0.6,
                             vmin=0, vmax=1, zorder=0)

plt.savefig(output_file, transparent=True)


