import pandas as pd, numpy as np
import config, os
from hepatotox_db import ANIMALDB
from stats import get_class_stats

import matplotlib.pyplot as plt

diseases = ['NECROSIS', 'CHOLESTASIS', 'STEATOSIS', 'DIRECT_DILI']

fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=plt.figaspect(3/2), sharex=True)

for i, d_name in enumerate(diseases):

    animals = pd.read_csv('../data_scratch/new_rat_data.csv', index_col=0)

    disease = animals[d_name]
    disease.index = animals.USUBJID

    txt_dir = os.path.join(config.TEXT_DIR, 'pls_results', d_name)
    img_dir = os.path.join(config.IMG_DIR, 'pls_results', d_name)

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    cv_predictions = pd.read_csv(os.path.join(txt_dir, 'cv_predictions.csv'), index_col=0)
    params = pd.read_csv(os.path.join(txt_dir, 'params.csv'), index_col=0)

    prediction_data = params.merge(cv_predictions)
    prediction_data.loc[prediction_data.PREDICTION < 0.5, 'PREDICTION_CLASS'] = 0
    prediction_data.loc[prediction_data.PREDICTION >= 0.5, 'PREDICTION_CLASS'] = 1


    stats = []
    for gp, gp_data in prediction_data.groupby(['MDL_ID', 'N_COMPONENTS']):
        stats_dic = get_class_stats(None, disease.loc[gp_data.USUBJID], gp_data.PREDICTION_CLASS)
        stats_dic['MDL_ID'] = gp[0]
        stats_dic['N_COMPONENTS'] = gp[1]
        stats_dic['ID'] = gp_data.ID.iloc[0]
        stats.append(stats_dic)

    stats_df = pd.DataFrame(stats)

    datasets = prediction_data.groupby('MDL_ID').ngroups
    n_components = prediction_data.groupby('N_COMPONENTS').ngroups

    stats_df['BAL_ACC'] = (stats_df['Recall'] + stats_df['Specificity']) / 2

    # fig, axes = plt.subplots(nrows=datasets, figsize=plt.figaspect(datasets/1))
    #
    # idx = 0
    # for gp, ds_stats in stats_df.groupby('MDL_ID'):
    #     axes[idx].scatter(ds_stats.N_COMPONENTS, ds_stats.BAL_ACC)
    #     axes[idx].plot(ds_stats.N_COMPONENTS, ds_stats.BAL_ACC)
    #     axes[idx].set_ylim(0.5, 1)
    #     idx = idx + 1
    #
    # plt.savefig(os.path.join(img_dir, 'n_cmps_stats_plt.png'))

    best_models = stats_df.groupby('MDL_ID').apply(lambda g: g[g['BAL_ACC'] == g['BAL_ACC'].max()].iloc[0])
    best_models.to_csv(os.path.join(txt_dir, 'best_models.csv'))
    best_model_predictions = prediction_data[prediction_data.ID.isin(best_models.ID)]

    best_model_predictions['PREDICTION_MEAN'] = best_model_predictions.groupby('USUBJID')['PREDICTION'].transform('mean')
    best_model_predictions['PREDICTION_STD'] = best_model_predictions.groupby('USUBJID')['PREDICTION'].transform('std')

    best_model_predictions.loc[best_model_predictions.PREDICTION_MEAN < 0.5, 'PREDICTION_MEAN_CLASS'] = 0
    best_model_predictions.loc[best_model_predictions.PREDICTION_MEAN >= 0.5, 'PREDICTION_MEAN_CLASS'] = 1

    final_data = best_model_predictions.drop_duplicates(['USUBJID', 'PREDICTION_MEAN_CLASS'])

    final_data = final_data.merge(disease.reset_index()).rename({0: 'TRUE'}, axis=1)

    offset = 0
    final_data = final_data[(final_data.PREDICTION_MEAN < 0.5-offset) | (final_data.PREDICTION_MEAN >= 0.5+offset)]



    bin_defs = np.arange(0, 1.1, 0.1)

    N, bins, patches = axarr[i, 0].hist(final_data.PREDICTION_MEAN[final_data[d_name] == 0], zorder=2, bins=bin_defs)

    for b, thispatch in zip(bins, patches):
        color = plt.cm.seismic(b)
        thispatch.set_facecolor(color)
        thispatch.set_edgecolor('black')

    N, bins, patches = axarr[i, 1].hist(final_data.PREDICTION_MEAN[final_data[d_name] == 1], zorder=2, bins=bin_defs)
    for b, thispatch in zip(bins, patches):
        color = plt.cm.seismic(b)
        thispatch.set_facecolor(color)
        thispatch.set_edgecolor('black')
    stats = get_class_stats(None, final_data[d_name], final_data.PREDICTION_MEAN_CLASS)

    n = final_data.shape[0]
    sens, spec, acc, ppv = stats['Recall'], stats['Specificity'], (stats['Recall']+stats['Specificity'])/2, stats['Precision']
    s = 'Sensitivity: {:.2f}%, Specificity: {:.2f}%, BalAcc: {:.2f}%, PPV: {:.2f}%, N: {}'.format(sens*100, spec*100, acc*100, ppv*100, n)

    axarr[i, 0].set_ylabel('Count')

    axarr[i, 0].set_title('{} Negative'.format(d_name.title()))
    axarr[i, 1].set_title('{} Positive'.format(d_name.title()))

    axarr[i, 0].set_xlabel('Prediction')
    axarr[i, 1].set_xlabel('Prediction')


    for idx in [0, 1]:
        axarr[i, idx].set_xlim(0, 1)
        axarr[i, idx].set_xticks(0, 1)
    # axarr[i].suptitle(s)

plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.savefig(os.path.join(config.IMG_DIR, 'average_model_preds.png'), transparent=True)
