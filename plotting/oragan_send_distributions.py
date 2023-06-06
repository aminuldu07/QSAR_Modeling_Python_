from send import send_db
import matplotlib.pyplot as plt

mi_data = send_db.generic_query('SELECT USUBJID, upper(MISTRESC) as RESULTS, upper(MISPEC) as ORGAN FROM MI')

mi_data['NORMAL'] = (mi_data.RESULTS == 'NORMAL') | (mi_data.RESULTS == 'UNREMARKABLE') | (mi_data.RESULTS == '') | (mi_data.RESULTS == ' ')

mi_data['N_RESULTS_ORGAN'] = mi_data.groupby('ORGAN')['RESULTS'].transform('count')
mi_data['N_NORMAL_ORGAN'] = mi_data.groupby('ORGAN')['NORMAL'].transform('sum')
mi_data['N_ANIMALS_ORGAN'] = mi_data.groupby('ORGAN')['USUBJID'].transform('nunique')
mi_data['N_ABNORMAL_ORGAN'] = mi_data['N_ANIMALS_ORGAN'] - mi_data['N_NORMAL_ORGAN']

mi_data = mi_data.drop_duplicates(['ORGAN', 'N_RESULTS_ORGAN', 'N_NORMAL_ORGAN', 'N_ABNORMAL_ORGAN']).sort_values('N_RESULTS_ORGAN', ascending=False).drop(['USUBJID', 'RESULTS'], axis=1)

top_ten = mi_data.iloc[:10]

fig, axarr = plt.subplots(nrows=1, ncols=top_ten.ORGAN.nunique(), figsize=plt.figaspect(1/top_ten.ORGAN.nunique()))
idx = 0
for organ, data in sorted(top_ten.groupby('ORGAN'), key=lambda x: x[1].N_ANIMALS_ORGAN.iloc[0], reverse=True):
    labels = ['ABNORMAL', 'NORMAL']
    sizes = [data.N_ABNORMAL_ORGAN.iloc[0], data.N_NORMAL_ORGAN.iloc[0]]
    n_animals = data.N_ANIMALS_ORGAN.iloc[0]
    wedges, _, _ = axarr[idx].pie(sizes, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=[(1, 0, 0, 0.6), (0, 1, 0, 0.6)])
    for w in wedges:
        w.set_linewidth(1)
        w.set_edgecolor('k')
    axarr[idx].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    title = '{}\nn={}'.format(organ.title(), n_animals)

    axarr[idx].set_title(organ.title())
    idx = idx + 1

import os, config
plt.savefig(os.path.join(config.IMG_DIR, 'send_pie_organs.png'), transparent=True)