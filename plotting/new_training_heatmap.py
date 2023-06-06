import pandas as pd
import matplotlib.pyplot as plt
import config,os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from joblib import dump, load
import math

def create_multimodels(y):
    actives = y[y == 1]
    inactives = shuffle(y[y == 0])

    splits = []

    for g, df in inactives.groupby(np.arange(len(inactives)) // actives.shape[0]):
        if df.shape[0] == actives.shape[0]:
            split_data = df.index.tolist() + actives.index.tolist()
        else:
            split_data = df.index.tolist() + actives.sample(df.shape[0]).index.tolist()
        splits.append(split_data)
    return splits

d_name = 'NECROSIS'

df = pd.read_csv('../data_scratch/new_rat_data.csv', index_col=0)
df = df.replace(np.inf, np.nan)
srted_tests = df.notnull().sum().sort_values(ascending=False)

good_tests = df.columns[(df.notnull().sum() / df.shape[0]) > 0.4]
good_tests = good_tests[~good_tests.isin(['USUBJID', 'STUDYID', 'SEX', 'STEATOSIS',
                                         'CHOLESTASIS', 'NECROSIS', 'SPECIES', 'IS_CONTROL',
                                         'BWDIFF', 'BWSLOPE', 'BWINTCEPT', 'MISTRESC'])]


data = df[good_tests]
data = data.apply(lambda x: x + abs(x.min()) + 1)
data = data.applymap(math.log10)


data.index = df.USUBJID
data['SEX'] = df['SEX']

le = LabelEncoder()
scaler = StandardScaler()


data['SEX'] = le.fit_transform(data['SEX'])

data = data.fillna(data.mean())

disease = df[d_name]
disease.index = df.USUBJID

train_predictions = []
cv_predictions = []
stats = []
id = 0
params = []

models = []

for mdl_idx, mdl_cmps in enumerate(create_multimodels(disease)):

    X = scaler.fit_transform(data.loc[mdl_cmps].values)
    y = disease.loc[mdl_cmps].values



    for n_cmps in list(range(1, 11)):

        pls = PLSRegression(n_components=n_cmps)
        pls.fit(X, y)

        preds = pls.predict(X)

        train_predictions = train_predictions + list(zip(mdl_cmps, preds[:, 0], [id]*len(mdl_cmps)))

        cv = model_selection.StratifiedKFold(shuffle=True, n_splits=10)



        for train, test in cv.split(X, y):

            train_X = X[train, :]
            train_y = y[train]

            test_X = X[test, :]
            test_y = y[test]

            pls_cv = PLSRegression(n_components=n_cmps)
            pls_cv.fit(train_X, train_y)

            test_preds = pls_cv.predict(test_X)
            cv_predictions = cv_predictions + list(zip(np.asarray(mdl_cmps)[test], test_preds[:, 0], [id]*len(test)))


        params.append((id, mdl_idx, n_cmps, ';'.join(mdl_cmps), ';'.join(data.columns.tolist())))
        models.append((id, pls))
        id = id + 1



preds_df = pd.DataFrame(train_predictions)

preds_df.columns = ['USUBJID', 'PREDICTION', 'ID']

cv_preds_df = pd.DataFrame(cv_predictions)

cv_preds_df.columns = ['USUBJID', 'PREDICTION', 'ID']

params_df = pd.DataFrame(params)

params_df.columns = ['ID', 'MDL_ID', 'N_COMPONENTS', 'TRAINING', 'FEATURES']



txt_dir = os.path.join(config.TEXT_DIR, 'pls_results', d_name)
if not os.path.exists(txt_dir):
    os.mkdir(txt_dir)

preds_df.to_csv(os.path.join(txt_dir, 'train_predictions_new_data.csv'))
cv_preds_df.to_csv(os.path.join(txt_dir, 'cv_predictions_new_data.csv'))
params_df.to_csv(os.path.join(txt_dir, 'params_new_data.csv'))

mdl_dir = os.path.join(config.MODEL_DIR, 'pls_results', d_name)
if not os.path.exists(mdl_dir):
    os.mkdir(mdl_dir)

for mdl_id, mdl in models:
    dump(mdl, os.path.join(mdl_dir, '{}_new_data.mdl'.format(mdl_id)))

