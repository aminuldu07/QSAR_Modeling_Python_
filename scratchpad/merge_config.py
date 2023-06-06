import pandas as pd


domains = ['CL', 'MA', 'MI', 'LB']

bw = pd.read_csv('../data_scratch/config_files/phuse/BWconfig.csv')

good_strains = bw.STRAIN.unique()

for dm in domains:
    data = pd.read_csv('../data_scratch/config_files/phuse/{}config.csv'.format(dm))
    data = data[data.STRAIN.isin(good_strains)]
    data.to_csv('../data_scratch/config_files/phuse/{}config.csv'.format(dm), index=False)