from experimental_animals import get_experimental_animals
import pandas as pd, numpy as np
import config
import matplotlib.pyplot as plt
import math
import os
from clincal_chemistry import get_clean_liver_clin_chem
from morphological_findings import get_classified_liver_results



def color_findings(findings):
    if findings['NECROSIS']:
        return 1, 0, 0, 1
    if findings['STEATOSIS']:
        return 0, 1, 1, 1
    if findings['CHOLESTASIS']:
        return 0, 0, 1, 1
    return 0, 1, 0, 1

animals = get_experimental_animals('RAT')
findings = get_classified_liver_results()
data = get_clean_liver_clin_chem()

data = pd.merge(data.reset_index(), animals[['USUBJID', 'SEX', 'MAXDOSE']])
data = pd.merge(data, findings)

data.set_index('USUBJID', inplace=True)

data = data[data[['SDH', 'BILI']].notnull().all(1)]

print(data.shape)

data['SDH_NORM'] = None
data['BILI_NORM'] = None

for sex in ['M', 'F']:
    data.loc[data.SEX == sex, 'SDH_NORM'] = data.SDH / config.lab_test_parameters['SDH'][sex]
    data.loc[data.SEX == sex, 'BILI_NORM'] = data.BILI / config.lab_test_parameters['BILI'][sex]


data.BILI_NORM = data.BILI_NORM.apply(lambda x: math.log10(x+1))
data.SDH_NORM = data.SDH_NORM.apply(lambda x: math.log10(x+1))




fig, ax = plt.subplots(1, 2, figsize=plt.figaspect(1/2)*2)

size = 60
csize = 30

output_file = os.path.join(os.getenv('LIVER_TOX'), 'img', '{}_mpl_edish_SDH_BILI_findings.png'.format('Rat'))


ax[0].plot([math.log10(3 + 1), math.log10(3 + 1)],
            [data['BILI_NORM'].min(), data['BILI_NORM'].max()], ls='--', color=(0/255, 0/255, 0/255, 1), zorder=3)

ax[0].plot([data['SDH_NORM'].min(), data['SDH_NORM'].max()],
           [math.log10(2 + 1), math.log10(2 + 1)],
           ls='--', color=(0/255, 0/255, 0/255, 1), zorder=3)

ax[0].scatter(data.loc[data.SEX == 'M', "SDH_NORM"], data.loc[data.SEX == 'M', "BILI_NORM"],
              facecolor=[color_findings(find) for i, find in data.loc[data.SEX == "M", ['STEATOSIS', 'CHOLESTASIS', 'NECROSIS']].iterrows()],
              edgecolor='black')

ax[0].set_xlabel('log((SDH / ULT)  + 1)')
ax[0].set_ylabel('log((BILI / ULT) + 1)')


ax[0].set_title('Male Rat')



# #
# #
# # FEMALE
# # DATA AND PLOT
# #
# #
#


ax[1].plot([math.log10(3+1), math.log10(3+1)],
            [data['BILI_NORM'].min(), data['BILI_NORM'].max()], ls='--', color=(0/255, 0/255, 0/255, 1), zorder=3)

ax[1].plot([data['SDH_NORM'].min(), data['SDH_NORM'].max()],
           [math.log10(2 + 1), math.log10(2 + 1)],
           ls='--', color=(0/255, 0/255, 0/255, 1), zorder=3)

ax[1].scatter(data.loc[data.SEX == "F", "SDH_NORM"], data.loc[data.SEX == "F", "BILI_NORM"],
              facecolor=[color_findings(find) for i, find in data.loc[data.SEX == "F", ['STEATOSIS', 'CHOLESTASIS', 'NECROSIS']].iterrows()],
              edgecolor='black')

ax[1].set_xlabel('log((SDH / ULT) + 1)')
#ax[1].set_ylabel('log(BILI / ULT + min(BILI / ULT) + 1)')

ax[1].set_title('Female {}'.format("Rat"))

tickfontsize = 18
titlefontsize = 24
labelfontsize = 22


# legend
ax[0].scatter([], [], s=size, facecolor=(1, 0, 0, 1), edgecolor='black', label='Necrosis')
ax[0].scatter([], [], s=size, facecolor=(0, 1, 1, 1), edgecolor='black', label='Steatosis')
ax[0].scatter([], [], s=size, facecolor=(0, 0, 1, 1), edgecolor='black', label='Cholestasis')
ax[0].scatter([], [], s=size, facecolor=(0, 0, 1, 1), edgecolor='black', label='Normal')
#
lg = ax[0].legend(title='Liver Findings:', fontsize=tickfontsize)
#
title = lg.get_title()
title.set_fontsize(tickfontsize)
#
#
#
#
# # ax[0].tick_params(color=(176/255, 224/255, 230/255, 1), labelcolor=(176/255, 224/255, 230/255, 1))
# # for spine in ax[0].spines.values():
# #     spine.set_edgecolor((176/255, 224/255, 230/255, 1))
# #
# #
# # ax[1].tick_params(color=(255/255, 192/255, 203/255, 1), labelcolor=(255/255, 192/255, 203/255, 1))
# # for spine in ax[1].spines.values():
# #     spine.set_edgecolor((255/255, 192/255, 203/255, 1))
#
#
#
#
for a in ax:
    a.tick_params(axis='x', labelsize=tickfontsize)
    a.tick_params(axis='y', labelsize=tickfontsize)
    a.set_title(a.get_title(), fontdict={'fontsize': titlefontsize})
    a.set_ylabel(a.get_ylabel(), fontsize=labelfontsize)
    a.set_xlabel(a.get_xlabel(), fontsize=labelfontsize)

plt.savefig(output_file, transparent=True)
#
