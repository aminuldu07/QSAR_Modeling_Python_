from send import send_db
import pandas as pd, numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from joblib import load

import matplotlib.pyplot as plt
import config, math, os
from collections import defaultdict

from matplotlib.ticker import NullFormatter

from clincal_chemistry import get_clean_liver_clin_chem
from morphological_findings import get_classified_liver_results
from experimental_animals import get_experimental_animals

# https://stackoverflow.com/questions/28256058/plotting-decision-boundary-of-logistic-regression

def color_findings(findings):
    if findings['NECROSIS']:
        return 1, 0, 0, 1
    if findings['STEATOSIS']:
        return 1, 1, 0, 1
    if findings['CHOLESTASIS']:
        return 1, 165/255, 0, 1
    return 0.8, 0.8, 0.8, 1

balanced = 'balanced'
for C in [100, 10, 1, 0.1, 0.01]:
    lr = LogisticRegression(C=C, class_weight=balanced)
    svc_linear = SVC(kernel='linear', class_weight=balanced,  C=C, probability=True)
    svc_poly = SVC(kernel='poly', class_weight=balanced,  C=C, probability=True)
    svc_rbf = SVC(kernel='rbf', class_weight=balanced,  C=C, probability=True)


    le_dictionary = defaultdict(LabelEncoder)


    animals = get_experimental_animals()

    chemistry_results = get_clean_liver_clin_chem()
    chemistry_results = chemistry_results[['ALT', 'BILI']]
    chemistry_results = chemistry_results[chemistry_results.notnull().all(1)]

    mi_results = get_classified_liver_results()

    full_data = pd.merge(chemistry_results.reset_index(), mi_results)
    full_data = pd.merge(full_data, animals)

    full_data = full_data[full_data[['ALT', 'BILI']].notnull().all(1)]

    full_data = full_data[full_data.SPECIES == 'RAT']

    y = full_data[['USUBJID', 'STEATOSIS', 'CHOLESTASIS', 'NECROSIS']]

    y.set_index('USUBJID', inplace=True)


    for test in ['ALT', 'BILI']:
        full_data.loc[full_data[test] < 0, test] = 0

    full_data.set_index('USUBJID', inplace=True)




    for mdl_name, mdl in [('lr', lr), ('svm_linear', svc_linear), ('svc_poly', svc_poly), ('svc_rbf', svc_rbf)]:

        data = full_data.copy()
        for sex in ['M', 'F']:
            data.loc[data.SEX == sex, 'ALT_NORM'] = data.ALT[data.SEX == sex] / config.lab_test_parameters['ALT'][sex]
            data.loc[data.SEX == sex, 'BILI_NORM'] = data.BILI[data.SEX == sex] / config.lab_test_parameters['BILI'][sex]

        scale_shift = 1

        data.BILI_NORM = data.BILI_NORM + scale_shift
        data.ALT_NORM = data.ALT_NORM + scale_shift

        data.BILI_NORM = data.BILI_NORM.apply(lambda x: math.log10(x))
        data.ALT_NORM = data.ALT_NORM.apply(lambda x: math.log10(x))



        X = data[['ALT_NORM', 'BILI_NORM', 'SEX']]

        # Encoding the variable
        X[['SEX']] = X[['SEX']].apply(lambda x: le_dictionary[x.name].fit_transform(x))
        # definitions for the axes

        mdl.fit(X, y.loc[X.index, 'NECROSIS'])

        for sex in ['M', 'F']:
            plot_data = data[data.SEX == sex]
            three_times_alt_norm = math.log10(
                (config.lab_test_parameters['ALT'][sex] * 3 / config.lab_test_parameters['ALT'][sex]))
            two_times_bili_norm = math.log10(
                (config.lab_test_parameters['BILI'][sex] * 2 / config.lab_test_parameters['BILI'][sex]))
            nullfmt = NullFormatter()  # no labels



            sex_colors = {'M': (176 / 255, 224 / 255, 230 / 255, 0.6),
                          'F': (255 / 255, 192 / 255, 203 / 255, 1)}

            sex_color = sex_colors[sex]

            sex_num = 1 if 'M' else 0


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
            xymax = np.max([np.max(np.fabs(plot_data.ALT_NORM.values)), np.max(np.fabs(plot_data.BILI_NORM))])
            lim = (int(xymax / binwidth) + 1) * binwidth

            # axScatter.set_xlim((-lim, lim))
            # axScatter.set_ylim((-lim, lim))

            # bins = np.arange(-lim, lim + binwidth, binwidth)


            x_heights, x_mids, _ = axHistx.hist(plot_data.ALT_NORM[plot_data.MAXDOSE == 0 & plot_data.CNTRL_GRP], bins=50, facecolor=sex_color, edgecolor='k')
            y_heights, y_mids, _ = axHisty.hist(plot_data.BILI_NORM[plot_data.MAXDOSE == 0 & plot_data.CNTRL_GRP], bins=50, facecolor=sex_color, edgecolor='k', orientation='horizontal')



            axScatter.scatter(plot_data.ALT_NORM[plot_data.MAXDOSE == 0 & plot_data.CNTRL_GRP],
                              plot_data.BILI_NORM[plot_data.MAXDOSE == 0 & plot_data.CNTRL_GRP], s=40,
                              facecolor=sex_color, edgecolor='k', zorder=2)


            axHistx.plot([three_times_alt_norm, three_times_alt_norm], [0, x_heights.max()], 'k--', zorder=1)
            axScatter.plot([three_times_alt_norm, three_times_alt_norm], [0, plot_data.BILI_NORM.max()], 'k--', zorder=1)

            axHisty.plot([0, y_heights.max()], [two_times_bili_norm, two_times_bili_norm], 'k--', zorder=1)
            axScatter.plot([0, plot_data.ALT_NORM.max()], [two_times_bili_norm, two_times_bili_norm], 'k--', zorder=1)

            axScatter.scatter(plot_data.ALT_NORM[plot_data.MAXDOSE != 0],
                              plot_data.BILI_NORM[plot_data.MAXDOSE != 0], s=75,
                              facecolor=[color_findings(find) for i, find in plot_data.loc[plot_data.MAXDOSE != 0, ['STEATOSIS', 'CHOLESTASIS', 'NECROSIS']].iterrows()],
                              edgecolor='k', zorder=1)


            axHistx.set_xlim(axScatter.get_xlim())
            axHisty.set_ylim(axScatter.get_ylim())

            axHistx.tick_params(labelsize=18)
            axHisty.tick_params(labelsize=18)
            axScatter.tick_params(labelsize=18)




            # handles, labels = axScatter.get_legend_handles_labels()
            #
            # axScatter.legend(handles[::-1], labels[::-1], fontsize=22, loc='best')

            axScatter.set_xlabel('log((ALT + min(ALT)+ 1 / ULT)', fontsize=22)
            axScatter.set_ylabel('log((BILI+ min(BILI) + 1 / ULT)', fontsize=22)


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






            #

            x_lims = axScatter.get_xlim()
            y_lims = axScatter.get_ylim()

            xx, yy = np.mgrid[x_lims[0]:x_lims[1]:0.01, y_lims[0]:y_lims[1]:0.01]

            #
            fake_data = np.c_[xx.ravel(),
                              yy.ravel(),
                              np.full(xx.ravel().shape, X.SEX.mean())]
            #
            probs = mdl.predict_proba(fake_data)[:, 1].reshape(xx.shape)



            #
            #




            #
            # # ax_c = f.colorbar(contour)
            # # ax_c.set_label("$P(Necrosis)$")
            #
            # ax[0].set_xlabel('log((ALT / ULT)  + 1)')
            # ax[0].set_ylabel('log((BILI / ULT) + 1)')
            #
            # ax[1].set_xlabel('log((ALT / ULT)  + 1)')
            #
            #
            # import os
            output_file = os.path.join(os.getenv('LIVER_TOX'), 'img', 'good_figures',
                                       'model_plot_{}_{}_{}_{}.png'.format(mdl_name, sex, C, balanced))

            # xx = xx-np.log10(2000)
            # yy = yy-np.log10(2000)

            # xx = alt_convert(xx)
            # yy = bili_convert(yy)

            contour = axScatter.contourf(xx, yy, probs, 25, cmap="RdBu_r", alpha=0.6,
                                  vmin=0, vmax=1, zorder=0)



            plt.savefig(output_file, transparent=True)

            plt.clf()

