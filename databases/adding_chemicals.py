import pandas as pd
from rdkit import Chem
import config
import numpy as np
from hepatotox_db import send_liver

compounds = [mol for mol in Chem.SDMolSupplier(config.CURATED_GSRS_FILE)]

for i, mol in enumerate(compounds):
    mol.SetProp('id', 'LIVERTOX_{}'.format(i+1))


all_columns = []

for mol in compounds:
    all_columns = all_columns + list(mol.GetPropNames())


all_columns = np.asarray(list(set(all_columns)))

merge_columns = all_columns[[col.startswith('Merge-Conflict-') for col in all_columns]]

rows = []

for mol in compounds:
    cols = list(mol.GetPropNames())
    cmp_data = {}
    for col in cols:
        try:
            cmp_data[col] = mol.GetProp(col)
        except UnicodeDecodeError as e:
            continue
    rows.append(cmp_data)

df = pd.DataFrame(rows)

for merge_col in merge_columns:
    reg_col = merge_col.replace('Merge-Conflict-', '').replace('.(1)', '')
    df.loc[df[reg_col].isnull(), reg_col] = df.loc[df[reg_col].isnull(), merge_col]

df['IS_ACTIVE_MOIETY_IN'] = [';'.join(list(set(apps.replace(' ', '').replace(',', ';').split(';'))))
                            for apps in df['IS_ACTIVE_MOIETY_IN']]

for i, row in df[df['IS_ACTIVE_INGREDIENT_IN'].notnull()].iterrows():
    apps = row['IS_ACTIVE_MOIETY_IN'].split(';')
    other_apps = list(set(row['IS_ACTIVE_INGREDIENT_IN'].replace(' ', '').replace(',', ';').split(';')))
    df.loc[i, 'IS_ACTIVE_MOIETY_IN'] = ';'.join(apps + other_apps)

df.to_csv('../data_scratch/cmp_data.csv')

apps_all = []

for row in df['IS_ACTIVE_MOIETY_IN']:
    apps_all = apps_all + row.split(';')


chem_data = []


for mol in compounds:
    if mol:
        mol_data = {}
        mol_data['MOL_BLOCK'] = Chem.MolToMolBlock(mol)
        mol_data['ID'] = mol.GetProp('id')
        chem_data.append(mol_data)

chem_data = pd.DataFrame(chem_data)



app_id = []

for i, row in df.iterrows():
    liver_id = row['id']
    bdnums = row['BDNUM']
    for app in row.IS_ACTIVE_MOIETY_IN.split(';'):
        app_id.append([liver_id, app, bdnums])

app_id = pd.DataFrame(app_id)

app_id = app_id.drop_duplicates()

app_id.columns = ['ID', 'APPNUMBER', 'BDNUMS']

# for app, data in app_id.groupby('APPNUMBER'):
#     if len(data) > 1:
#         print(len(data))
#         print(app, data)

db = send_liver.connect_to_liverdb()

chem_data.to_sql('CHEMICALS', con=db)
app_id.to_sql('APPID', con=db)

db.close()