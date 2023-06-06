from send import send_db
import pandas as pd, numpy as np
import config
import matplotlib.pyplot as plt
import math
import os
from clincal_chemistry import get_clean_liver_clin_chem, \
    filter_text, hepatobiliary_lab_tests, hepatocellular_lab_tests, valid_lab_test_units
from scipy.stats import norm, lognorm
from experimental_animals import get_experimental_animals

dm = send_db.generic_query('SELECT STUDYID, USUBJID, ARMCD, SETCD, SPECIES, SEX FROM DM')
ex = send_db.generic_query('SELECT STUDYID, USUBJID, EXTRT, MAX(EXDOSE) as MAXDOSE '
                           'FROM EX GROUP BY STUDYID, USUBJID')


data = get_clean_liver_clin_chem()

animal_data = get_experimental_animals('RAT')

control_rats = animal_data[((animal_data.MAXDOSE == 0) & animal_data.CNTRL_GRP)][['USUBJID', 'SEX']]

males = control_rats[control_rats.SEX == 'M'].USUBJID
females = control_rats[control_rats.SEX == 'F'].USUBJID

data = pd.merge(data.reset_index(), control_rats[['USUBJID', 'SEX']])

data.set_index('USUBJID', inplace=True)

configuration_frame = pd.DataFrame(columns=['M', 'F'])


fig, axarr = plt.subplots(nrows=2, ncols=4, figsize=plt.figaspect(2/4)*2)
for i, test in enumerate(hepatocellular_lab_tests):

    # log normal distrubtions for male
    male_values = data.loc[males, test].dropna()

    if test in ['AST']:
        bins = np.arange(0, math.ceil(data[test].max())+1, 2)
    elif test in ['SDH-']:
        bins = np.arange(0, math.ceil(data[test].max())+1, 0.5)
    else:
        bins = np.arange(0, math.ceil(data[test].max()) + 1, 1)

    male_norm_heights, mids, _ = axarr[0, i].hist(male_values, density=True, bins=bins,
                                               facecolor=(176/255, 224/255, 230/255, 0.6), edgecolor='k',
                                               #label='Male Distribution'
                                                  )

    min_x = mids.min()
    max_x = mids.max()

    x_range = np.linspace(min_x, max_x, 1000)
    shape, loc, scale = lognorm.fit(male_values + 1, floc=0)
    p = lognorm.pdf(x_range, shape, scale=scale)
    axarr[0, i].plot(x_range, p, c=(176/255, 224/255, 230/255, 1), linewidth=2)
    m_max_95 = lognorm.ppf(0.95, shape, scale=scale)

    print("{} Male: {}".format(test, m_max_95))

    axarr[0, i].plot([m_max_95, m_max_95], [0, male_norm_heights.max()],
                  c=(176/255, 224/255, 230/255, 1), ls='--', label='M: {:.2f}'.format(m_max_95))

    # log normal distrubtions for female
    female_values = data.loc[females, test].dropna()
    female_norm_heights, mids, _ = axarr[0, i].hist(female_values, density=True,
                                                 bins=bins, facecolor=(255/255, 192/255, 203/255, 0.6), edgecolor='k',
                                                 #label='Female Distribution'
                                                    )
    min_x = mids.min()
    max_x = mids.max()
    x_range = np.linspace(min_x, max_x, 1000)

    shape, loc, scale = lognorm.fit(female_values + 1, floc=0)
    p = lognorm.pdf(x_range, shape, scale=scale)
    axarr[0, i].plot(x_range, p, c=(255/255, 192/255, 203/255, 1), linewidth=2)

    f_max_95 = lognorm.ppf(0.95, shape, scale=scale)

    print("{} Female: {}".format(test, f_max_95))

    axarr[0, i].plot([f_max_95, f_max_95],
                  [0, female_norm_heights.max()],
                  c=(255/255, 192/255, 203/255, 1),
                  ls='--', label='F: {:.2f}'.format(f_max_95))

    axarr[0, i].legend(loc='upper right', fontsize=14)

    axarr[0, i].set_xlim([0, f_max_95 + 10])

    axarr[0, i].set_title(test)
    if i == 0:
        axarr[0, i].set_ylabel('Frequency', fontsize=22)

    configuration_frame.loc[test] = [m_max_95, f_max_95]

for i, test in enumerate(hepatobiliary_lab_tests):
    # log normal distrubtions for male
    data.loc[data[test] < 0, test] = 0

    male_values = data.loc[males, test].dropna()

    if test in ['ALP']:
        bins = np.arange(0, math.ceil(data[test].max())+1, 4)
    elif test in ['BILI']:
        bins = np.arange(0, math.ceil(data[test].max())+1, 0.5)
    else:
        bins = np.arange(0, math.ceil(data[test].max()) + 1, 1)


    male_norm_heights, mids, _ = axarr[1, i].hist(male_values, density=True, bins=bins,
                                               facecolor=(176/255, 224/255, 230/255, 0.6), edgecolor='k',
                                               #label='Male Distribution'
                                                  )

    min_x = mids.min()
    max_x = mids.max()

    x_range = np.linspace(min_x, max_x, 1000)
    shape, loc, scale = lognorm.fit(male_values + 1, floc=0)
    p = lognorm.pdf(x_range, shape, scale=scale)
    axarr[1, i].plot(x_range, p, c=(176/255, 224/255, 230/255, 1), linewidth=2)
    m_max_95 = lognorm.ppf(0.95, shape, scale=scale)

    print("{} Male: {}".format(test, m_max_95))

    axarr[1, i].plot([m_max_95, m_max_95], [0, male_norm_heights.max()],
                  c=(176/255, 224/255, 230/255, 1), ls='--', label='M: {:.2f}'.format(m_max_95))

    # log normal distrubtions for female
    female_values = data.loc[females, test].dropna()
    female_norm_heights, mids, _ = axarr[1, i].hist(female_values, density=True,
                                                 bins=bins, facecolor=(255/255, 192/255, 203/255, 0.6), edgecolor='k',
                                                 #label='Female Distribution'
                                                    )
    min_x = mids.min()
    max_x = mids.max()
    x_range = np.linspace(min_x, max_x, 1000)

    shape, loc, scale = lognorm.fit(female_values + 1, floc=0)
    p = lognorm.pdf(x_range, shape, scale=scale)
    axarr[1, i].plot(x_range, p, c=(255/255, 192/255, 203/255, 1), linewidth=2)

    f_max_95 = lognorm.ppf(0.95, shape, scale=scale)

    print("{} Female: {}".format(test, f_max_95))

    axarr[1, i].plot([f_max_95, f_max_95],
                  [0, female_norm_heights.max()],
                  c=(255/255, 192/255, 203/255, 1),
                  ls='--', label='F: {:.2f}'.format(f_max_95))

    axarr[1, i].legend(loc='upper right', fontsize=14)

    axarr[1, i].set_xlim([0, max(m_max_95, f_max_95) + 10])

    axarr[1, i].set_title(test)

    if i == 0:
        axarr[1, i].set_ylabel('Frequency', fontsize=22)

    configuration_frame.loc[test] = [m_max_95, f_max_95]

all_tests = hepatocellular_lab_tests + hepatobiliary_lab_tests
for ax, tst in zip(axarr.flatten(), all_tests):
    ax.set_xticks([round(x, 2) for x in np.linspace(0, ax.get_xticks()[-1], 5)])
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)

    ax.set_yticks([round(y, 4) for y in np.linspace(0, ax.get_yticks()[-1], 5)])
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)

    ax.set_title(ax.get_title(), fontsize=22)

    ax.set_xlabel(valid_lab_test_units[tst], fontsize=16)

configuration_frame.to_csv(os.path.join(config.TEXT_DIR, 'lab_test_parameters.csv'))
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.savefig(os.path.join(config.IMG_DIR, 'lab_tests.png'), transparent=True)
