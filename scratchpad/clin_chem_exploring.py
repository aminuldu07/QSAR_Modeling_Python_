from send import send_db
import json

lb = send_db.generic_query('SELECT USUBJID, LBSTRESN, LBSTRESC, '
                           'LBSTRESU, '
                           'upper(LBTEST) as LBTEST, '
                           'upper(LBTESTCD) as LBTESTCD, '
                           'upper(LBSPEC) as LBSPEC, '
                           'upper(LBCAT) as LBCAT, '
                           'VISITDY '
                           'FROM LB')

lb_hema = lb[lb.LBCAT == 'HEMATOLOGY']

test_in_category = {}

test_in_category['HEMATOLOGY'] = {}

for tst, tst_data in lb_hema.groupby('LBTESTCD'):
    print(tst, tst_data.LBSTRESU.value_counts())

    test_in_category['HEMATOLOGY'][tst] = (tst_data.LBSTRESU.value_counts() / tst_data.LBSTRESU.shape[0]).map(lambda n: '{:.2%}'.format(n)).to_dict()


lb_urine = lb[(lb.LBCAT == 'URINALYSIS') | (lb.LBCAT == 'URINALYSIS/URINE CHEMISTRY')]

test_in_category['URINALYSIS;URINALYSIS/URINE CHEMISTRY'] = {}

for tst, tst_data in lb_urine.groupby('LBTESTCD'):
    print(tst, tst_data.LBSTRESU.value_counts())

    test_in_category['URINALYSIS;URINALYSIS/URINE CHEMISTRY'][tst] = (tst_data.LBSTRESU.value_counts() / tst_data.LBSTRESU.shape[0]).map(lambda n: '{:.2%}'.format(n)).to_dict()



with open('../data_scratch/lb_test_units_2.json', 'w') as f:
    json.dump(test_in_category, f, indent=4, sort_keys=True)