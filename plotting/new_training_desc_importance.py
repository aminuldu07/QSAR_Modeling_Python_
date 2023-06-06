import pandas as pd
import config, os
from hepatotox_db import ANIMALDB
from stats import get_class_stats

import seaborn as sns
import matplotlib.pyplot as plt

species = 'RAT'
d_name = 'CHOLESTASIS'


animals = pd.read_csv('../data_scratch/new_rat_data.csv', index_col=0).drop('STUDYID', axis=1)


disease = animals[d_name]
disease.index = animals.USUBJID
animals = animals.set_index('USUBJID')
txt_dir = os.path.join(config.TEXT_DIR, 'pls_results', d_name)
img_dir = os.path.join(config.IMG_DIR, 'pls_results', d_name)

if not os.path.exists(img_dir):
    os.mkdir(img_dir)

cv_predictions = pd.read_csv(os.path.join(txt_dir, 'train_predictions_new_data.csv'), index_col=0)
params = pd.read_csv(os.path.join(txt_dir, 'params_new_data.csv'), index_col=0)

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
best_models.to_csv(os.path.join(txt_dir, 'best_models_new_data.csv'))
best_model_predictions = prediction_data[prediction_data.ID.isin(best_models.ID)]

best_model_predictions['PREDICTION_MEAN'] = best_model_predictions.groupby('USUBJID')['PREDICTION'].transform('mean')
best_model_predictions['PREDICTION_STD'] = best_model_predictions.groupby('USUBJID')['PREDICTION'].transform('std')

best_model_predictions.loc[best_model_predictions.PREDICTION_MEAN < 0.5, 'PREDICTION_MEAN_CLASS'] = 0
best_model_predictions.loc[best_model_predictions.PREDICTION_MEAN >= 0.5, 'PREDICTION_MEAN_CLASS'] = 1

final_data = best_model_predictions.drop_duplicates(['USUBJID', 'PREDICTION_MEAN_CLASS'])

final_data = final_data.merge(disease.reset_index()).rename({0: 'TRUE'}, axis=1)

offset = 0.2
final_data = final_data[(final_data.PREDICTION_MEAN < 0.5-offset) | (final_data.PREDICTION_MEAN > 0.5+offset)]



false_positives = final_data[(final_data.PREDICTION_MEAN_CLASS == 1) & (final_data[d_name] == 0)]
false_negatives = final_data[(final_data.PREDICTION_MEAN_CLASS == 0) & (final_data[d_name] == 1)]
true_positives = final_data[(final_data.PREDICTION_MEAN_CLASS == 1) & (final_data[d_name] == 1)]
true_negatives = final_data[(final_data.PREDICTION_MEAN_CLASS == 0) & (final_data[d_name] == 0)]

X_data = animals.loc[:, animals.columns != 'SEX']

x_melt = X_data.loc[:, X_data.columns != 'SEX'].reset_index().melt(id_vars='USUBJID')

x_melt.loc[x_melt.USUBJID.isin(false_negatives.USUBJID), 'CLASS'] = 'FN'
x_melt.loc[x_melt.USUBJID.isin(false_positives.USUBJID), 'CLASS'] = 'FP'
x_melt.loc[x_melt.USUBJID.isin(true_negatives.USUBJID), 'CLASS'] = 'TN'
x_melt.loc[x_melt.USUBJID.isin(true_positives.USUBJID), 'CLASS'] = 'TP'


sorted_tp_fp = abs((X_data.loc[true_negatives.USUBJID].median() - X_data.loc[true_positives.USUBJID].median())).sort_values().iloc[-10:]


# fig, ax = plt.subplots()
#
# for gp, name in zip([false_positives, false_negatives, true_positives, true_negatives], ['FP', 'FN', 'TP', 'TN']):
#     ax.plot(X_data.loc[gp.USUBJID].median().index, X_data.loc[gp.USUBJID].median(), label=name)
# ax.legend()
# plt.show()
# fig, ax = plt.subplots(figsize=plt.figaspect(1/1))
# sns.boxplot(x='variable', y='value', hue='CLASS', data=x_melt, ax=ax)

# g = sns.catplot(x="CLASS", y="value",
#                 hue="CLASS", col="variable",
#                 data=x_melt, kind="box", col_wrap=4, sharey=False, showfliers = False)
#
# plt.savefig(os.path.join(config.IMG_DIR, 'pls_results', d_name, 'boxplot.png'))

x_melt = x_melt[(x_melt.CLASS.isin(['TP', 'TN'])) & (x_melt.variable.isin(sorted_tp_fp.index))]
x_melt.value = x_melt.value.astype('float')
sns.violinplot(x="variable", y="value", hue="CLASS",
               split=True, inner="quart",
               data=x_melt)
plt.show()