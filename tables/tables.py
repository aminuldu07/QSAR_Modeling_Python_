"""

Module script containing caching for the table
data in the manuscript.

"""
from send import send_db
from config import TEXT_DIR
import os

tables_dir = os.path.join(TEXT_DIR, 'manuscript')
if not os.path.exists(tables_dir):
    os.mkdir(tables_dir)
tables_dir = os.path.join(TEXT_DIR, 'manuscript', 'tables')
if not os.path.exists(tables_dir):
    os.mkdir(tables_dir)

# Table 1.
# Frequency counts of liver histopath results
# for rats in SEND

animals = send_db.get_all_animals()
rats = animals[animals.SPECIES == 'RAT']

liver_results = send_db.generic_query('SELECT USUBJID, upper(MISTRESC) as RESULTS '
                                      'FROM MI WHERE MISPEC == "LIVER"')

liver_results.RESULTS.value_counts().to_csv(os.path.join(tables_dir,
                                                         'table_1_liver_histopath_counts.csv'))