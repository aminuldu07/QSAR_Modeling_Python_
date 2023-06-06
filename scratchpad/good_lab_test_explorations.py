import sqlite3 as sql
from send import send_db
import pandas as pd, numpy as np
from hepatotox_db import filter_sacrifice_phase
from morphological_findings import get_classified_liver_results
from clincal_chemistry import filter_text
from scipy import stats

species = 'RAT'

# get a list of animals from the SEND DB
# and filter based off of species
animals = send_db.get_all_animals()
target_animals = animals[animals.SPECIES == species]

# the modeling set should only contain
# terminal animals (i.e., not recovery animals)
# for that, we use the function filter_sacrifice_phase

target_animals = filter_sacrifice_phase(target_animals, phase='treatment')

# add the specific set codes for these animals
setcodes = send_db.generic_query('SELECT STUDYID, USUBJID, SETCD FROM DM')
target_animals = target_animals.merge(setcodes, on=['STUDYID', 'USUBJID'])

# need to now go through each group by study and identify
# the control animals.  We only include animals that
# are in a study with a negative control.

tx = send_db.generic_query("SELECT STUDYID, SETCD, TXVAL FROM TX WHERE TXPARMCD == 'TCNTRL'")
target_animals = target_animals.merge(tx, on=['STUDYID', 'SETCD'], how='left')

good_animals = []

standAlonesWords = ["placebo", "untreated", "sham"]
currentModifiers = ["negative", "saline", "peg", "vehicle", "citrate", "dextrose", "water", "air"]
control_expression = r'|'.join(standAlonesWords + currentModifiers)

for study, data in target_animals.groupby('STUDYID'):
    if (data.TXVAL.value_counts().shape[0] >= 1) and (
    data.TXVAL.str.contains(control_expression, case=False, na=False).any()):
        good_animals.append(data)

good_animals = pd.concat(good_animals)

good_animals.loc[good_animals.TXVAL.str.contains(control_expression, case=False, na=False), 'IS_CONTROL'] = True
good_animals.loc[~good_animals.TXVAL.str.contains(control_expression, case=False, na=False), 'IS_CONTROL'] = False

lb = send_db.generic_query('SELECT STUDYID, '
                           'USUBJID, '
                           'LBSTRESC, '
                           'LBTESTCD, '
                           'upper(LBCAT) as LBCAT, '
                           'upper(LBSCAT) as LBSCAT, '
                           'upper(LBSPEC) as LBSPEC, '
                           'LBSTRESU FROM LB')
lb = lb[lb.USUBJID.isin(good_animals.USUBJID)]
lb.loc[:, 'LBSTRESC'] = lb.LBSTRESC.apply(filter_text)


bad_tests = []

# cache tests with more than one
# unit
for gp, gp_data in lb.groupby(['STUDYID', 'LBTESTCD', 'LBSPEC']):
    if (gp_data.LBSTRESU.value_counts().shape[0] > 1) and (not (gp_data.LBSTRESU == '').any()):
        print(gp)
        print(gp_data.LBSTRESU.value_counts())
        print(gp_data.USUBJID.nunique())
        bad_tests.append(gp_data)

pd.concat(bad_tests).to_csv('../data_scratch/rat_tests_multiple_units.csv')

lb = lb[lb.LBSTRESC.notnull()]

lb['LBSTRESC_MAX'] = lb.groupby(['STUDYID', 'USUBJID',  'LBTESTCD', 'LBSPEC'])['LBSTRESC'].transform('max')

max_responses = lb.drop_duplicates(['STUDYID', 'USUBJID',  'LBTESTCD', 'LBSPEC', 'LBSTRESC_MAX'])

max_responses.loc[max_responses.LBSPEC == '', 'LBSPEC'] = 'UNSPECIFIED'

max_responses['LBTESTCD_SPEC'] = max_responses.LBTESTCD + '-' + max_responses.LBSPEC

converted_tests = []


for gp, gp_data in max_responses.groupby(['STUDYID', 'LBTESTCD_SPEC']):

    # if there are more than one unit
    # for a test, take the unit thats most populated
    if (gp_data.LBSTRESU.value_counts().shape[0] > 1) and (not (gp_data.LBSTRESU == '').any()):

        best_unit = gp_data.LBSTRESU.value_counts().index[0]

        unit_data = gp_data[gp_data.LBSTRESU == best_unit]
        converted_tests.append(unit_data)
    else:
        converted_tests.append(gp_data)


converted_tests = pd.concat(converted_tests)

animals_pivot = converted_tests.pivot_table(index=['STUDYID', 'USUBJID'], columns='LBTESTCD_SPEC', values='LBSTRESC_MAX').reset_index()
del animals_pivot.columns.name

mi = get_classified_liver_results()


tests = animals_pivot.columns[~animals_pivot.columns.isin(['STUDYID', 'USUBJID', 'SPECIES', 'SEX', 'IS_CONTROL',
                                                           'NECROSIS', 'CHOLESTASIS', 'STEATOSIS'])]

animals_with_mi = animals_pivot.merge(mi, how='left').merge(good_animals[['STUDYID', 'USUBJID', 'SPECIES', 'SEX', 'IS_CONTROL']])

for disease in ['NECROSIS', 'CHOLESTASIS', 'STEATOSIS']:
    animals_with_mi.loc[animals_with_mi[disease].isnull(), disease] = 0

for study, data in animals_with_mi.groupby(['STUDYID', 'SEX']):
    control_animals_mean = data[data.IS_CONTROL][tests].mean()
    animals_with_mi.loc[data.index, tests] = data[tests].divide(control_animals_mean)


bw = send_db.generic_query('SELECT STUDYID, USUBJID, BWSTRESN, BWSTRESU, BWTESTCD, BWDY FROM BW')
animals_bw = animals_with_mi[['STUDYID', 'USUBJID', 'IS_CONTROL', 'SEX']].merge(bw)
# animals_bw = animals_bw[animals_bw.BWSTRESU != '']

# identify control animals for normalization
animals_bw['IS_CONTROL'] = animals_bw['IS_CONTROL'].astype(bool)

# these are the functions that will actually create
# the features from body weight data
def difference_fx(data):
    first_weight = data.sort_values(by='BWDY')['BWSTRESN'].iloc[0]
    second_weight = data.sort_values(by='BWDY')['BWSTRESN'].iloc[-1]
    return second_weight - first_weight

def slope_fx(data):

    slope, intercept, r_value, p_value, std_err = stats.linregress(data.BWDY, data.BWSTRESN)
    return slope

def intercept_fx(data):

    slope, intercept, r_value, p_value, std_err = stats.linregress(data.BWDY, data.BWSTRESN)
    return intercept

animals_bw_diff = animals_bw.groupby('USUBJID').apply(difference_fx).reset_index()
animals_bw_diff.columns = ['USUBJID', 'BWDIFF']

animals_bw = animals_bw.merge(animals_bw_diff)

animals_bw_slope = animals_bw.groupby('USUBJID').apply(slope_fx).reset_index()
animals_bw_slope.columns = ['USUBJID', 'BWSLOPE']

animals_bw = animals_bw.merge(animals_bw_slope)

animals_bw_int = animals_bw.groupby('USUBJID').apply(intercept_fx).reset_index()
animals_bw_int.columns = ['USUBJID', 'BWINTCEPT']

animals_bw = animals_bw.merge(animals_bw_int)

new_tests = ['BWDIFF', 'BWSLOPE', 'BWINTCEPT']
names = list(map(lambda x: '{}_NORM'.format(x), new_tests))

for name in names:
    animals_bw[name] = np.nan

animals_bw = animals_bw[['STUDYID', 'USUBJID', 'IS_CONTROL', 'SEX'] + new_tests + names].drop_duplicates()
animals_bw.index = animals_bw.USUBJID
for study, data in animals_bw.groupby(['STUDYID', 'SEX']):
    control_animals_mean = data[data.IS_CONTROL][new_tests].mean()

    for name in names:
        animals_bw.loc[data.index, name] = data[name.replace('_NORM', '')] / control_animals_mean[
            name.replace('_NORM', '')]

final_data = animals_with_mi.merge(animals_bw.reset_index(drop=True).drop(['IS_CONTROL', 'SEX'], axis=1), how='left', on=['STUDYID', 'USUBJID'])
final_data.to_csv('../data_scratch/new_rat_data.csv')