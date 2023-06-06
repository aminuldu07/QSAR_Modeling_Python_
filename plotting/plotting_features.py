from hepatotox_db import send_liver, send_db
from classic_ml import GridSearchModel, RandomForestClassifier
from converter import ClinicalChemistry, Hematology
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import time, copy

import os, config


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

np.random.seed(2)

debug = False
iterations = 25
start = time.time()


rat_data = send_liver.generic_query('SELECT * FROM TERMINALRAT')
rat_data.index = rat_data.USUBJID
rat_data = rat_data.replace(np.inf, np.nan)


liv_enzyme = ['ALT', 'AST', 'BILI', 'ALP'] + ['SEX']
all_clin_chem = rat_data.columns[rat_data.columns.isin(ClinicalChemistry.TESTS.keys())]
all_hem = rat_data.columns[rat_data.columns.isin(Hematology.TESTS.keys())]

good_tests_cc = all_clin_chem[(rat_data[all_clin_chem].notnull().sum() > 10000)].tolist() + ['SEX'] + ['BWDIFF_NORM', 'BWSLOPE_NORM', 'BWINTCEPT_NORM']
good_tests_hem = all_hem[(rat_data[all_hem].notnull().sum() > 10000)].tolist() + ['SEX'] + ['BWDIFF_NORM', 'BWSLOPE_NORM', 'BWINTCEPT_NORM']

full_features = list(set(good_tests_cc + good_tests_hem))

feature_sets = [liv_enzyme,
                good_tests_cc,
                good_tests_hem,
                full_features]

classes = ['CHOLESTASIS', 'STEATOSIS', 'NECROSIS']

if debug:
    _, rat_data, _, _ = model_selection.train_test_split(rat_data, rat_data['NECROSIS'],
                                                     stratify=rat_data['NECROSIS'], test_size=0.1)


stats = []
predictions = []
params = []

id = 1
models = []

for features in feature_sets:
    feature_data = rat_data[features + classes]
    feature_data = feature_data[feature_data.notnull().all(1)]

