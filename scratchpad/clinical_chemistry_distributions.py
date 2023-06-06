from send import send_db
import pandas as pd
from converter import ClinicalChemistry
from clincal_chemistry import filter_text, is_number
from scipy.stats import norm


acceptable_strains = ['BEAGLE', 'CYNOMOLGUS', 'BALB/C', 'C57BL/6', 'CB6F1-TGN (RASH2)', 'CD1(ICR)'
              'GOTTINGEN', 'NEW ZEALAND', 'SPRAGUE-DAWLEY', 'WISTAR HAN']

animals = send_db.get_control_animals()
animals = animals[animals.STRAIN.isin(acceptable_strains)]

final_data = []

for tst in list(ClinicalChemistry.TESTS.keys()):
    clinical_chemistry_results = send_db.generic_query('SELECT USUBJID, LBSTRESN, LBSTRESC, LBSTRESU, LBTEST, VISITDY FROM LB WHERE upper(LBCAT) == ? '
                                                       'and upper(LBTESTCD) == ?', ('CLINICAL CHEMISTRY', tst))

    print(clinical_chemistry_results.LBTEST.iloc[0])

    lb_results = clinical_chemistry_results[clinical_chemistry_results.LBSTRESN.notnull()]
    lb_results = lb_results[lb_results.LBSTRESU.isin(ClinicalChemistry.TESTS[tst].keys())]

    def convert(row):
        return ClinicalChemistry.TESTS[tst][row['LBSTRESU']] * row['LBSTRESN']

    units = None
    for unit, val in ClinicalChemistry.TESTS[tst].items():
        if val == 1:
            units = unit
            break

    if lb_results.shape[0] > 100:

        lb_results['LBSTRESN_CONVERTED'] = lb_results.apply(convert, axis=1)

        lb_results['LBSTRESN_CONVERTED_MEAN'] = lb_results.groupby('USUBJID')['LBSTRESN_CONVERTED'].transform('mean')

        for gp, data in animals.groupby(['SPECIES', 'STRAIN', 'SEX']):
            test_data = data.merge(lb_results, on='USUBJID')

            lb_result_data = test_data.LBSTRESN[test_data.LBSTRESN.notnull()]
            n = lb_result_data.shape[0]
            if n != 0:
                mu, std = norm.fit(test_data.LBSTRESN)
                final_data.append(['CLINICAL CHEMISTRY', clinical_chemistry_results.LBTEST.iloc[0], tst] + list(gp) + [round(mu, 2), round(std, 2), unit, n])

columns = ['LBCAT', 'LBTEST', 'LBTESTCD', 'SPECIES', 'STRAIN', 'SEX', 'LBSTRESM', 'LBSTRESSD', 'LBSTRESU', 'N']

df = pd.DataFrame(final_data, columns=columns)

df.to_csv('../data_scratch/config_files/LBconfig.csv', index=None)