"""
Throw away script to just cache all the different clinical chemistry
endpoints and their different usnits.

"""

from send import send_db
import json

#lb_categories = send_db.generic_query('SELECT DISTINCT upper(LBCAT) FROM LB')['upper(LBCAT)'].values.tolist()

lb_categories = ['CLINICAL CHEMISTRY']

test_in_category = {}


for lb_cat in lb_categories:
    lb_tests = send_db.generic_query('SELECT DISTINCT upper(LBTESTCD) FROM LB WHERE upper(LBCAT) == ?',
                                     query_params=(lb_cat,))['upper(LBTESTCD)'].values.tolist()
    lb_test_dic = {}
    for lb_tst in lb_tests:
        units = send_db.generic_query('SELECT LBSTRESU FROM LB WHERE upper(LBCAT) == ? AND upper(LBTESTCD) == ?',
                                         query_params=(lb_cat, lb_tst))['LBSTRESU']

        lb_test_dic[lb_tst] = (units.value_counts() / units.shape[0]).map(lambda n: '{:.2%}'.format(n)).to_dict()

    test_in_category[lb_cat] = lb_test_dic

with open('../data_scratch/lb_test_units.json', 'w') as f:
    json.dump(test_in_category, f, indent=4, sort_keys=True)
