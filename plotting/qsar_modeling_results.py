from hepatotox_db import send_liver
from morphological_findings import is_steatosis, is_necrosis, is_cholestasis
from send import send_db
import math, numpy as np, pandas as pd
from stats import get_class_stats



stats = send_liver.generic_query('SELECT * from QSAR_STATS')
params = send_liver.generic_query('SELECT * from QSAR_PARAMS')
stats = stats.merge(params)

mdl='rf'
cv='10'
percent = 0.55
new_stats = stats.copy()
new_stats = new_stats[((new_stats.MODEL_TYPE == mdl) &
                       (new_stats.CV == cv) &
                       (new_stats.FEATURES == 'ecfp6') &
                       (new_stats.MINOR_CLASS_PERCENT == percent))]



diseases = new_stats['DISEASE'].unique()

disease_frame_avg = []
disease_frame_std = []

for disease in diseases:
    disease_stats = new_stats[new_stats.DISEASE == disease]
    rows = []
    for iteration in range(1, 16):
        iteration_stats = disease_stats[disease_stats.ITER == iteration]

        stats = dict([(met, val)
                          for met, val in zip(iteration_stats.METRIC, iteration_stats.VALUE)])
        stats['BalAcc'] = (stats['Recall'] + stats['Specificity']) / 2
        rows.append(stats)

    df = pd.DataFrame(rows)
    df.loc[:, ~df.columns.isin(['No. Positive', 'No. Negative'])] = df.loc[:, ~df.columns.isin(
        ['No. Positive', 'No. Negative'])].applymap(lambda val: round(val * 100, 2))
    df = df.reset_index()
    df.columns = ['Disease'] + df.columns.tolist()[1:]
    avg = df.mean()
    std = df.std()
    avg['Disease'] = disease
    std['Disease'] = disease

    disease_frame_avg.append(avg)
    disease_frame_std.append(std)

features_avg = pd.concat(disease_frame_avg, axis=1).T
features_std = pd.concat(disease_frame_std, axis=1).T

features_avg.to_csv('../data_scratch/{}_avg.csv'.format('QSAR'))
features_std.to_csv('../data_scratch/{}_std.csv'.format('QSAR'))


#
#
# for features, name in pretty_features_dic.items():
#
#     mdl='rf'
#     cv='10'
#     percent = 0.55
#     new_predictions = predictions.copy()
#     new_predictions = new_predictions[((new_predictions.MODEL_TYPE == mdl) &
#                                        (new_predictions.CV == cv) &
#                                        (new_predictions.FEATURES == features) &
#                                        (new_predictions.MINOR_CLASS_PERCENT == percent))]
#
#     diseases = new_predictions['DISEASE'].unique()
#
#
#
#     disease_frame_avg = []
#     disease_frame_std = []
#
#     for disease in diseases:
#
#         disease_predictions = new_predictions[new_predictions.DISEASE == disease]
#
#         rows = []
#
#
#         for iteration in range(1, 26):
#             iter_predictions = disease_predictions[disease_predictions.ITER == iteration]
#             iter_predictions = iter_predictions[['USUBJID', 'PREDICTION']].drop_duplicates()
#             iter_predictions = iter_predictions[iter_predictions.PREDICTION.notnull()]
#
#             iter_predictions = iter_predictions.merge(data[[disease, 'USUBJID']], on=['USUBJID'])
#
#             acts = (iter_predictions[disease] == 1).sum()
#             inacts = (iter_predictions[disease] == 0).sum()
#
#             stats = get_class_stats(None, iter_predictions[disease], (iter_predictions.PREDICTION >= 0.5).astype(int))
#             stats['BalAcc'] = (stats['Recall'] + stats['Specificity']) / 2
#             stats['No. Positive'] = acts
#             stats['No. Negative'] = inacts
#             rows.append(stats)
#
#         df = pd.DataFrame(rows)
#         df.loc[:, ~df.columns.isin(['No. Positive', 'No. Negative'])] = df.loc[:, ~df.columns.isin(
#             ['No. Positive', 'No. Negative'])].applymap(lambda val: round(val * 100, 2))
#         df = df.reset_index()
#         df.columns = ['Disease'] + df.columns.tolist()[1:]
#         avg = df.mean()
#         std = df.std()
#         avg['Disease'] = disease
#         std['Disease'] = disease
#
#         disease_frame_avg.append(avg)
#         disease_frame_std.append(std)
#
#     features_avg = pd.concat(disease_frame_avg, axis=1).T
#     features_std = pd.concat(disease_frame_std, axis=1).T
#
#     features_avg.to_csv('../data_scratch/{}_avg.csv'.format(name))
#     features_std.to_csv('../data_scratch/{}_std.csv'.format(name))