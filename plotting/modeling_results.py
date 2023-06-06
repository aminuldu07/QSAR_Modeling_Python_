from hepatotox_db import send_liver
from morphological_findings import is_steatosis, is_necrosis, is_cholestasis
from send import send_db
import math, numpy as np, pandas as pd
from stats import get_class_stats



data = send_liver.generic_query('SELECT * FROM TERMINALRAT')
predictions = send_liver.generic_query('SELECT * from PREDICTIONS')
params = send_liver.generic_query('SELECT * from PARAMS')
predictions = predictions.merge(params)


pretty_features_dic = {
    'ALT,AST,BILI,ALP,SEX': 'Liver Enzymes',
    'ALB,ALBGLOB,ALP,ALT,AST,BILI,CA,CHOL,CK,CL,CREAT,GGT,GLOBUL,GLUC,K,PHOS,PROT,SODIUM,TRIG,UREAN,SEX,BWDIFF_NORM,BWSLOPE_NORM,BWINTCEPT_NORM': 'Clinical Chemistry',
    'BASO,BASOLE,EOS,EOSLE,HCT,HGB,LGUNSCE,LYM,LYMLE,MCH,MCHC,MCV,MONO,MONOLE,NEUT,NEUTLE,PLAT,RBC,RDW,RETI,RETIRBC,WBC,SEX,BWDIFF_NORM,BWSLOPE_NORM,BWINTCEPT_NORM': 'Hematology',
    'RBC,ALBGLOB,BWSLOPE_NORM,CK,K,TRIG,BWINTCEPT_NORM,PROT,PHOS,UREAN,MCV,MCHC,GGT,RETIRBC,BASO,CHOL,LYM,BILI,NEUTLE,'
    'PLAT,CREAT,EOSLE,ALP,RETI,LYMLE,HGB,AST,SEX,MONOLE,ALT,BASOLE,BWDIFF_NORM,EOS,SODIUM,WBC,NEUT,LGUNSCE,HCT,CA,ALB,'
    'MONO,GLOBUL,RDW,CL,MCH,GLUC': 'All'
}







for features, name in pretty_features_dic.items():

    mdl='rf'
    cv='10'
    percent = 0.55
    new_predictions = predictions.copy()
    new_predictions = new_predictions[((new_predictions.MODEL_TYPE == mdl) &
                                       (new_predictions.CV == cv) &
                                       (new_predictions.FEATURES == features) &
                                       (new_predictions.MINOR_CLASS_PERCENT == percent))]

    diseases = new_predictions['DISEASE'].unique()



    disease_frame_avg = []
    disease_frame_std = []

    for disease in diseases:

        disease_predictions = new_predictions[new_predictions.DISEASE == disease]

        rows = []


        for iteration in range(1, 26):
            iter_predictions = disease_predictions[disease_predictions.ITER == iteration]
            iter_predictions = iter_predictions[['USUBJID', 'PREDICTION']].drop_duplicates()
            iter_predictions = iter_predictions[iter_predictions.PREDICTION.notnull()]

            iter_predictions = iter_predictions.merge(data[[disease, 'USUBJID']], on=['USUBJID'])

            acts = (iter_predictions[disease] == 1).sum()
            inacts = (iter_predictions[disease] == 0).sum()

            stats = get_class_stats(None, iter_predictions[disease], (iter_predictions.PREDICTION >= 0.5).astype(int))
            stats['BalAcc'] = (stats['Recall'] + stats['Specificity']) / 2
            stats['No. Positive'] = acts
            stats['No. Negative'] = inacts
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

    features_avg.to_csv('../data_scratch/{}_avg.csv'.format(name))
    features_std.to_csv('../data_scratch/{}_std.csv'.format(name))

# for disease in disease:
#     disease_predictions = new_predictions[new_predictions.DISEASE == disease]
#
#
#     disease_predictions = disease_predictions[['USUBJID', 'AVG', disease]].drop_duplicates()
#     disease_predictions = disease_predictions[disease_predictions.AVG.notnull()]




    # stats = get_class_stats(None, disease_predictions[disease], (disease_predictions.AVG >= threshold).astype(int))
    # stats.pop('AUC')
    # xs = []
    # ys = []
    # stats['BalAcc'] = (stats['Recall'] + stats['Specificity']) / 2
    #
    # stats['No. Positive'] = acts
    # stats['No. Negative'] = inacts
    # rows.append(stats)
    # for stat, metric in stats.items():
    #
    #     if 'No.' not in stat:
    #         xs.append(stat)
    #         ys.append(metric)
    #
    #
    # pred_hist.update_yaxes(type="log", row=row + 1, col=1)
    #
    # df = pd.DataFrame(rows)
    # df.index = diseases
    # df.loc[:, ~df.columns.isin(['No. Positive', 'No. Negative'])] = df.loc[:, ~df.columns.isin(
    #     ['No. Positive', 'No. Negative'])].applymap(lambda val: round(val * 100, 2))
    # df = df.reset_index()
    # df.columns = ['Disease'] + df.columns.tolist()[1:]
    #
    # acts = df.pop('No. Positive')
    # inacts = df.pop('No. Negative')
    #
    # df.insert(1, 'No. Positive', acts)
    # df.insert(1, 'No. Negative', inacts)
    #
    # table = dash_table.DataTable(
    #     data=df.to_dict("rows"),
    #     columns=[{"name": i, "id": i} for i in df.columns],
    # )