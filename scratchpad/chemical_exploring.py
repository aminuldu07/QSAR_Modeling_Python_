from hepatotox_db import send_liver
from send import send_db
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from classic_ml import GridSearchModel, RandomForestClassifier
from sklearn import model_selection


def calc_ecfp6(molecules, name_col='CASRN'):
    """
    Takes in a list of rdkit molecules and returns ECFP6 fingerprints for a list of rdkit molecules
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecule objects with no None values
    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    data = []

    for mol in molecules:
        ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024)]
        data.append(ecfp6)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])

def calc_class_weights(y, ratios=[0.45/0.55, 0.5/0.5, 0.55/0.45]):
    major_class = y.value_counts().sort_values(ascending=False).index[0]
    minor_class = y.value_counts().sort_values(ascending=False).index[1]
    num_major_class = (y == major_class).sum()
    num_minor_class = (y == minor_class).sum()

    class_weights = []
    for ratio in ratios:
        total_needed = int(ratio * num_major_class)
        minor_class_weight = total_needed / num_minor_class
        class_weights.append({minor_class: minor_class_weight,
                              major_class: 1})
    return class_weights



chemicals = send_liver.generic_query('SELECT * FROM CHEMICALS')
chem_apps = send_liver.generic_query('SELECT * FROM APPID')

chemicals = chemicals.merge(chem_apps, on='ID')
chemicals = chemicals.rename({'ID': 'TOXID'}, axis=1)

rats = send_liver.generic_query('SELECT * FROM TERMINALRAT')
applications = send_db.generic_query('SELECT * FROM AN')

ex = send_db.generic_query('SELECT USUBJID, EXDOSE, EXDOSU FROM EX')

applications = applications[applications.APPNUMBER.apply(lambda num: num[:3] == 'IND')]

rats = rats.merge(applications[['STUDYID', 'APPNUMBER']], on=['STUDYID'])

rats_chemicals = rats.merge(chemicals, on=['APPNUMBER'])

diseases = ['NECROSIS', 'CHOLESTASIS', 'STEATOSIS']
tests = ['ALT', 'AST', 'BILI', 'ALP']

rows = []

for app, data in rats_chemicals.groupby('APPNUMBER'):

    if data['TOXID'].unique().shape[0] > 1:
        continue

    control_data = data[data.IS_CONTROL.astype('bool')]
    noncontrol_data = data[~data.IS_CONTROL.astype('bool')]

    row_data = {}
    row_data['APPNUMBER'] = app
    row_data['TOXID'] = data['TOXID'].unique()[0]

    for disease in diseases:
        control_percent = (control_data[disease] == 1).sum() / control_data.shape[0]
        noncontrol_percent = (noncontrol_data[disease] == 1).sum() / noncontrol_data.shape[0]
        if ((control_percent + 0.1) < noncontrol_percent):
            row_data[disease] = 1
        else:
            row_data[disease] = 0

    for test in tests:

        fold_increase = noncontrol_data[test].mean()

        if np.isnan(fold_increase):
            row_data[test] = np.nan
        elif fold_increase > 1.5:
            row_data[test] = 1
        else:
            row_data[test] = 0
    rows.append(row_data)

app_data = pd.DataFrame(rows).drop_duplicates(subset='TOXID').merge(chemicals[['TOXID', 'MOL_BLOCK']], on='TOXID')

db = send_liver.connect_to_liverdb()

app_data.to_sql('QSAR_TRAINING', con=db, if_exists='replace')

db.close()

mols = []


for id, mb in zip(app_data.TOXID, app_data.MOL_BLOCK):
    mol = Chem.MolFromMolBlock(mb)
    mol.SetProp('TOXID', id)
    mols.append(mol)

fps = calc_ecfp6(mols, 'TOXID').reset_index().drop_duplicates().set_index('index')

iterations = 15

stats = []
predictions = []
models = []
params = []

id = 1

for disease in ['NECROSIS', 'STEATOSIS', 'ALT', 'AST']:

    y = app_data[['TOXID', disease]]
    y = y.set_index('TOXID')[disease]

    y = y[y.notnull()]
    X = fps.loc[y.index]

    ratios = [0.45 / 0.55, 0.5 / 0.5, 0.55 / 0.45]
    class_weights_list = calc_class_weights(y, ratios=ratios)

    for minor_class_percent, class_weight in zip([0.45, 0.5, 0.55], class_weights_list):

        for iteration in range(1, iterations + 1):
            rf = RandomForestClassifier(max_depth=10,  # max depth 10 to prevent overfitting
                                        class_weight='balanced',
                                        random_state=0)
            rf_params = {'n_estimators': [5, 10, 25, 50, 100]}
            model = GridSearchModel(rf, 'rf', rf_params)

            model.model.class_weight = class_weight
            model.direct_fit(X, y)
            stats_dic, probas = model.get_stats(X, y.values, return_preds='probas')

            pred_frame = pd.DataFrame(probas, index=X.index, columns=[id])
            predictions.append(pred_frame)

            stats_frame = pd.DataFrame(stats_dic.values(), index=stats_dic.keys(), columns=[id])

            stats.append(stats_frame)

            param = [id, disease, 'None', iteration, model.name, minor_class_percent, 'ecfp6']
            params.append(param)

            models.append((id, model.tuned_model))
            id = id + 1

        splits = [10]
        for n_splits in splits:

            for iteration in range(1, iterations + 1):
                cv = model_selection.StratifiedKFold(shuffle=True, n_splits=n_splits)
                rf = RandomForestClassifier(max_depth=10,  # max depth 10 to prevent overfitting
                                            class_weight='balanced',
                                            random_state=0)
                rf_params = {'n_estimators': [5, 10, 25, 50]}
                model = GridSearchModel(rf, 'rf', rf_params)
                model.model.class_weight = class_weight
                model.cross_validate(X, y, cv)
                five_fold_stats, probas = model.get_cv_stats(X, y.values, cv, return_preds='probas')

                pred_frame = pd.DataFrame(probas, index=X.index, columns=[id])
                predictions.append(pred_frame)

                stats_frame = pd.DataFrame(five_fold_stats, index=five_fold_stats.keys(), columns=[id])
                stats.append(stats_frame)

                param = [id, disease, n_splits, iteration, model.name, minor_class_percent, 'ecfp6']
                params.append(param)

                models.append((id, model.tuned_model))
                id = id + 1

import pickle, time

params = pd.DataFrame(params, columns=['ID', 'DISEASE', 'CV',
                                       'ITER', 'MODEL_TYPE',
                                       'MINOR_CLASS_PERCENT',
                                       'FEATURES'])
stats = pd.concat(stats, axis=1)
#predictions = pd.concat(predictions, axis=1)
models_frame = pd.DataFrame(models, columns=['ID', 'MODEL'])
models_frame.MODEL = models_frame.MODEL.apply(lambda x: pickle.dumps(x))

# predictions = pd.melt(predictions.reset_index(), id_vars=['index']).rename(
#     {"index": "USUBJID", 'variable': 'ID', 'value': 'PREDICTION'}, axis=1)
stats = pd.melt(stats.reset_index(), id_vars=['index']).rename({'index': 'METRIC', 'variable': 'ID', 'value': 'VALUE'},
                                                               axis=1)

db = send_liver.connect_to_liverdb()
# predictions.to_sql('qsar_predictions', db, if_exists='replace', index=False)
stats.to_sql('qsar_stats', db, if_exists='replace', index=False)
params.to_sql('qsar_params', db, if_exists='replace', index=False)
models_frame.to_sql('qsar_models', db, dtype={'MODEL': 'BLOB'}, if_exists='replace', index=False)


db.close()
