"""
Script to add the original dose received as an integer value
in an effort to capture dose when concatenate.

The algorithm should proceed as follows:

1) Identify all studies with a chemical structure.
2) Eliminate those with multiple structures.
3) Identify studies that have a singular unique treatment
in EXTRT
4) Make sure these studies have three different dose groups


"""

from hepatotox_db import send_liver
from send import send_db
import pandas as pd, numpy as np

# get studies with structures
studies_with_structure = send_liver.generic_query('SELECT * FROM APPID').merge(send_db.generic_query('SELECT * FROM AN'))

print("{} applications have structures".format(studies_with_structure.APPNUMBER.nunique()))

# remove studies with more than 1 compounds attributed to them
studies_with_structure['N_CMPS'] = studies_with_structure.groupby('APPNUMBER')['ID'].transform('nunique')
studies_with_structure = studies_with_structure[studies_with_structure.N_CMPS == 1]

print("{} applications have only one structure".format(studies_with_structure.APPNUMBER.nunique()))

# now elimate any study that has more than one treatment

ex = send_db.generic_query('SELECT STUDYID, USUBJID, EXDOSE, upper(EXTRT) as TRTMNT FROM EX').merge(studies_with_structure)
ex['N_TRTMNT'] = ex.groupby('APPNUMBER')['TRTMNT'].transform('nunique')
ex = ex[ex['N_TRTMNT'] == 1]

print("{} applications have only one structure".format(ex.APPNUMBER.nunique()))


# what needs to be done is identify studies that only have a
# control, low medium and high group

training_set_studies = send_liver.generic_query('SELECT STUDYID, USUBJID FROM TERMINALRAT')

ex = ex.merge(training_set_studies)

dm = send_db.generic_query('SELECT STUDYID, USUBJID, SETCD FROM DM')
tx = send_db.generic_query('SELECT STUDYID, SETCD, TXVAL as TXARMCD FROM TX WHERE TXPARMCD == "ARMCD"')

ex['MAXDOSE'] = ex.groupby(['STUDYID', 'USUBJID'])['EXDOSE'].transform('max')
ex = ex.drop_duplicates(['USUBJID', 'MAXDOSE'])


data = dm.merge(ex, how='left').merge(tx, on=['STUDYID', 'SETCD'])
data = data[data.STUDYID.isin(ex.STUDYID)]

data['SETMEAN'] = data.groupby(['STUDYID', 'TXARMCD'])['MAXDOSE'].transform('mean')

data = data.fillna(0)

bad_studies = []

data['DOSE_GRP'] = np.nan
data['DOSE_FLAG'] = np.nan

# identify dose groups within a study
# if dose groups can be easily broken
# up into control, low, med, high dose
# easily (e.g., using the average MAX
# dose from EXDOSE with SETS) if not
# try and merge groups into three
# sets using the numpy's np.cut, which
# will split doses into three bins of
# equal ranges.

for gp, study_data in data.groupby('STUDYID'):

    controls = study_data[study_data.SETMEAN == 0]
    noncontrols = study_data[study_data.SETMEAN != 0]
    if not noncontrols.empty:

        if noncontrols.SETMEAN.value_counts().shape[0] == 3:
            # print(noncontrols.SETMEAN.rank(method='dense'), noncontrols.SETMEAN.value_counts())
            data.loc[controls.index, 'DOSE_GRP'] = 0
            data.loc[noncontrols.index, 'DOSE_GRP'] = noncontrols.SETMEAN.rank(method='dense')
            data.loc[controls.index, 'DOSE_FLAG'] = False
            data.loc[noncontrols.index, 'DOSE_FLAG'] = False

        elif noncontrols.SETMEAN.value_counts().shape[0] > 3:
            try:
                assignments = pd.cut(noncontrols.SETMEAN, bins=3, labels=False) + 1
                data.loc[controls.index, 'DOSE_GRP'] = 0
                data.loc[noncontrols.index, 'DOSE_GRP'] = assignments.values
                print(study_data.STUDYID.iloc[0])
                print(noncontrols.TXARMCD.unique())
                print(noncontrols.SETMEAN.value_counts())
                data.loc[controls.index, 'DOSE_FLAG'] = True
                data.loc[noncontrols.index, 'DOSE_FLAG'] = True
            except ValueError as e:
                # print(e)
                # print(assignments.value_counts(), noncontrols.SETMEAN.value_counts())
                bad_studies.append(study_data.STUDYID.iloc[0])


conn = send_liver.connect_to_liverdb()

if 'DOSE_GRP' not in send_liver.generic_query('PRAGMA table_info(TERMINALRAT);').name.values:
    query_string = 'ALTER TABLE TERMINALRAT ADD COLUMN DOSE_GRP'
    conn.execute(query_string)

if 'DOSE_FLAG' not in send_liver.generic_query('PRAGMA table_info(TERMINALRAT);').name.values:
    query_string = 'ALTER TABLE TERMINALRAT ADD COLUMN DOSE_FLAG'
    conn.execute(query_string)

for i, rats in data[['DOSE_GRP', 'DOSE_FLAG', 'USUBJID']].iterrows():
    usubjid = rats['USUBJID']
    dosegrp = rats['DOSE_GRP']
    doseflag = rats['DOSE_FLAG']

    if not np.isnan(dosegrp):
        dosegrp = int(dosegrp)

    qs = "UPDATE TERMINALRAT SET DOSE_GRP = ? WHERE USUBJID = ?"
    conn.execute(qs, (dosegrp, usubjid))

    qs = "UPDATE TERMINALRAT SET DOSE_FLAG = ? WHERE USUBJID = ?"
    conn.execute(qs, (doseflag, usubjid))

conn.commit()
conn.close()
