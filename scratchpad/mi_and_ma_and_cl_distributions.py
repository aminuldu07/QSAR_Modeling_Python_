from send import send_db
import pandas as pd
from converter import ClinicalChemistry
from config import CONTROLLED_TERM_FILE


controlled_terms = pd.read_csv(CONTROLLED_TERM_FILE, sep='\t')

mi_codelist = ['C120531', 'C88025', 'C132321']

mi_controlled_terms = controlled_terms[controlled_terms['Codelist Code'].isin(mi_codelist)]['CDISC Submission Value'].str.upper()

spec_list = ['C77529']
spec_controlled_terms = controlled_terms[controlled_terms['Codelist Code'].isin(spec_list)]['CDISC Submission Value'].str.upper()

acceptable_strains = ['BEAGLE', 'CYNOMOLGUS', 'BALB/C', 'C57BL/6', 'CB6F1-TGN (RASH2)', 'CD1(ICR)'
              'GOTTINGEN', 'NEW ZEALAND', 'SPRAGUE-DAWLEY', 'WISTAR HAN']

animals = send_db.get_control_animals()
animals = animals[animals.STRAIN.isin(acceptable_strains)]

mi = send_db.generic_query('SELECT STUDYID, USUBJID, upper(MISPEC) as MISPEC, upper(MISTRESC) as MISTRESC FROM MI WHERE upper(MITESTCD) == "GHISTXQL"')

mi = mi[mi.MISTRESC.isin(mi_controlled_terms)]
mi = mi[mi.MISPEC.isin(spec_controlled_terms)]
mi.MISTRESC = mi.MISTRESC.str.replace('NORMAL', 'UNREMARKABLE')
merged_data = animals.merge(mi, on=['STUDYID', 'USUBJID'])

final_data = []

for grp, data in merged_data.groupby(['SPECIES', 'STRAIN', 'SEX', 'MISPEC']):
    counts = data.MISTRESC.value_counts().rename('MIFREQ').reset_index().rename({'index':'MISTRESC'}, axis=1)
    dummy_data = pd.DataFrame([list(grp) for _ in range(counts.shape[0])], columns=['SPECIES', 'STRAIN', 'SEX', 'MISPEC'])
    counts['N'] = [counts.MIFREQ.sum() for _ in range(counts.shape[0])]
    counts.MIFREQ = counts.MIFREQ / counts.MIFREQ.sum()
    final_data.append(pd.concat([dummy_data, counts], axis=1))

pd.concat(final_data).to_csv('../data_scratch/config_files/MIconfig.csv', index=False)

ma = send_db.generic_query('SELECT STUDYID, USUBJID, upper(MASPEC) as MASPEC, upper(MASTRESC) as MASTRESC FROM MA WHERE upper(MATESTCD) == "GROSPATH"')
ma.MASTRESC = ma.MASTRESC.str.replace('NORMAL', 'UNREMARKABLE')
ma = ma[ma.MASPEC.isin(spec_controlled_terms)]
merged_data = animals.merge(ma, on=['STUDYID', 'USUBJID'])

final_data = []

for grp, data in merged_data.groupby(['SPECIES', 'STRAIN', 'SEX', 'MASPEC']):
    counts = data.MASTRESC.value_counts().rename('MAFREQ').reset_index().rename({'index': 'MASTRESC'}, axis=1)
    dummy_data = pd.DataFrame([list(grp) for _ in range(counts.shape[0])],
                              columns=['SPECIES', 'STRAIN', 'SEX', 'MASPEC'])
    counts['N'] = [counts.MAFREQ.sum() for _ in range(counts.shape[0])]
    counts.MAFREQ = counts.MAFREQ / counts.MAFREQ.sum()
    final_data.append(pd.concat([dummy_data, counts], axis=1))

pd.concat(final_data).to_csv('../data_scratch/config_files/MAconfig.csv', index=False)


cl = send_db.generic_query('SELECT STUDYID, USUBJID, upper(CLTEST) as CLTEST, upper(CLTESTCD) as CLTESTCD, '
                           'upper(CLSTRESC) as CLSTRESC FROM CL WHERE upper(CLCAT) == "CLINICAL SIGNS"')


cl.CLSTRESC = cl.CLSTRESC.str.replace('NORMAL', 'UNREMARKABLE')

merged_data = animals.merge(cl, on=['STUDYID', 'USUBJID'])

final_data = []

for grp, data in merged_data.groupby(['SPECIES', 'STRAIN', 'SEX', 'CLTEST', 'CLTESTCD']):
    counts = data.CLSTRESC.value_counts().rename('CLFREQ').reset_index().rename({'index':'CLSTRESC'}, axis=1)
    dummy_data = pd.DataFrame([list(grp) for _ in range(counts.shape[0])], columns=['SPECIES', 'STRAIN', 'SEX', 'CLTEST', 'CLTESTCD'])
    counts['N'] = [counts.CLFREQ.sum() for _ in range(counts.shape[0])]
    counts.CLFREQ = counts.CLFREQ / counts.CLFREQ.sum()
    final_data.append(pd.concat([dummy_data, counts], axis=1))

pd.concat(final_data).to_csv('../data_scratch/config_files/CLconfig.csv', index=False)