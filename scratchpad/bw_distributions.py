from send import send_db
import pandas as pd

from config import CONTROLLED_TERM_FILE
from scipy.stats import norm

acceptable_strains = ['BEAGLE', 'CYNOMOLGUS', 'BALB/C', 'C57BL/6', 'CB6F1-TGN (RASH2)', 'CD1(ICR)'
              'GOTTINGEN', 'NEW ZEALAND', 'SPRAGUE-DAWLEY', 'WISTAR HAN']

animals = send_db.get_control_animals()
animals = animals[animals.STRAIN.isin(acceptable_strains)]

bw = send_db.generic_query('SELECT STUDYID, USUBJID, upper(BWTESTCD) as BWTESTCD, BWSTRESN, BWSTRESU, VISITDY FROM BW')
bw = bw[((bw.BWSTRESU != '') & (bw.BWSTRESN != '') & (bw.BWSTRESN.notnull()))]

bw.BWSTRESN[bw.BWSTRESU == 'kg'] = bw.BWSTRESN[bw.BWSTRESU == 'kg'] / 1000

terminal_weights = bw[bw.BWTESTCD == 'TERMBW']
bw = bw[bw.BWTESTCD == 'BW']

# get first body weights
first_weights = bw[bw.groupby('USUBJID')['VISITDY'].transform(lambda row: row == row.min())]

final_data = []

for gp, data in animals.groupby(['SPECIES', 'STRAIN', 'SEX']):

    if data.shape[0] > 90:
        bw_data = data.merge(first_weights, on='USUBJID')
        term_bw_data = data.merge(terminal_weights, on='USUBJID')

        n = bw_data.shape[0]

        mu, std = norm.fit(bw_data.BWSTRESN)
        final_data.append(['Body Weight', 'BW'] + list(gp) + [round(mu, 4), round(std, 4), 'g', n])



        n = term_bw_data.shape[0]
        mu, std = norm.fit(term_bw_data.BWSTRESN)
        final_data.append(['Terminal Body Weight', 'TERMBW'] + list(gp) + [round(mu, 4), round(std, 4), 'g', n])

df = pd.DataFrame(final_data, columns=['BWTEST', 'BWTESTCD', 'SPECIES', 'STRAIN', 'SEX', 'BWSTRESM', 'BWSTRESSD', 'BWSTRESU', 'N'])

df.to_csv('../data_scratch/config_files/BWconfig.csv', index=False)