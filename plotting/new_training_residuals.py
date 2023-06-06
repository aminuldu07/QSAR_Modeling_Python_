from hepatotox_db import ANIMALDB, send_liver
import config, os
from rdkit import Chem
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np, pandas as pd
import seaborn as sns
from joblib import load
from stats import get_class_stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math

species = 'RAT'
d_name = 'CHOLESTASIS'

txt_dir = os.path.join(config.TEXT_DIR, 'pls_results', d_name)
mdl_dir = os.path.join(config.MODEL_DIR, 'pls_results', d_name)
img_dir = os.path.join(config.IMG_DIR, 'pls_results', d_name)

if not os.path.exists(img_dir):
    os.mkdir(img_dir)
best_models = pd.read_csv(os.path.join(txt_dir, 'best_models.csv'), index_col=0)
params = pd.read_csv(os.path.join(txt_dir, 'params_new_data.csv'), index_col=0).set_index('ID')

best_estimators = {}

for mdl_id in best_models.ID:
    best_estimators[mdl_id] = load(os.path.join(mdl_dir, '{}_new_data.mdl'.format(int(mdl_id))))



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


data['SEX'] = le.fit_transform(df['SEX'])

data = data.fillna(data.mean())

disease = df[d_name]
disease.index = df.USUBJID

sum_of_sum_of_squares = []

for ID, pls in best_estimators.items():

    training = params.loc[ID, 'TRAINING'].split(';')
    features = params.loc[ID, 'FEATURES'].split(';')

    X = scaler.fit_transform(data.loc[training, features].values)
    y = disease.loc[training]
    T = pls.x_scores_
    P = pls.x_loadings_

    E = X - np.matmul(T, P.T)

    ss = pd.Series((E*E).sum(0), index=data.columns)
    sum_of_sum_of_squares.append(ss/ss.sum())


df = pd.concat(sum_of_sum_of_squares, axis=1)
df.columns = ['mdl-{}'.format(col) for col in df.columns]

df['AVG'] = df.mean(1)

df.sort_values('AVG', inplace=True)

fig, axarr = plt.subplots(figsize=(20, 7.5), ncols=1, nrows=1)
sns.heatmap(df, cmap=sns.color_palette("RdBu", 7),
            cbar=False, square=True, linecolor='black', linewidths=0.25, ax=axarr, cbar_kws={"orientation": "horizontal"})

axarr.set_yticks([x+0.5 for x in np.arange(len(df))])


axarr.set_yticklabels(df.index, rotation=0, fontsize=10)
axarr.set_xticklabels(df.columns, rotation=90, fontsize=11)

plt.savefig(os.path.join(img_dir, 'feature_heatmap_new_data.png'), transparent=True)

top_five = (sum(sum_of_sum_of_squares) / len(sum_of_sum_of_squares)).sort_values().iloc[:5].index

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=plt.figaspect(1/4))

for i, test in enumerate(top_five[1:]):

    # col = np.where(data.columns == test)[0][0]
    # first = np.where(data.columns == top_five[0])[0][0]
    #
    # training = params.loc[2, 'TRAINING'].split(';')
    # features = params.loc[2, 'FEATURES'].split(';')
    #
    # XXX = data.loc[training, features]
    # YYY = disease[training]
    #
    # ax[i].scatter(XXX.values[YYY.values == 0, col], XXX.values[YYY.values == 0, first], color='blue')
    # ax[i].scatter(XXX.values[YYY.values == 1, col], XXX.values[YYY.values == 1, first], color='red')


    ax[i].scatter(data.loc[disease == 0, test], data.loc[disease == 0, top_five[0]], color='blue')
    ax[i].scatter(data.loc[disease == 1, test], data.loc[disease == 1, top_five[0]], color='red')

    ax[i].set_xlabel(test)

ax[0].set_ylabel(top_five[0])

plt.savefig(os.path.join(img_dir, 'top5_full_new_data.png'), transparent=True)

desc_ranked = (sum(sum_of_sum_of_squares) / len(sum_of_sum_of_squares)).sort_values()

desc_ranked.to_csv(os.path.join(txt_dir, 'top_desc_new_data.csv'))