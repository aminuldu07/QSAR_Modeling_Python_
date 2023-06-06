from hepatotox_db import ANIMALDB
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import IMG_DIR
import os, math

figures_dir = os.path.join(IMG_DIR, 'manuscript')
if not os.path.exists(figures_dir):
    os.mkdir(figures_dir)

figures_dir = os.path.join(IMG_DIR, 'manuscript', 'figures')
if not os.path.exists(figures_dir):
    os.mkdir(figures_dir)

rats = ANIMALDB('RAT')

sex = 'F'


diseases = ['NECROSIS']

lbtests = ['ALB', 'RBC', 'HCT', 'WBC', 'HGB']

clin_chem = rats.get_clin_chem(min_response_perc=0.6)
sex = rats.get_sexes()
bws = rats.get_body_weights()

data = clin_chem.join(sex, how='inner').join(bws, how='inner')

disease = rats.get_liver_disease('NECROSIS')
disease = disease.loc[data.index]

rats.db.index = rats.db.USUBJID

df = rats.db.loc[data.index, diseases+lbtests]

fig, axarr = plt.subplots(ncols=5,
                          nrows=1,
                          sharex=True,
                          figsize=plt.figaspect(1/5))

swarm = True

for i, lbtest in enumerate(lbtests):

    data = []
    lbtestdata = df[df[lbtest].notnull()]

    data.append(lbtestdata[lbtestdata['NECROSIS'] == 0][lbtest].values)

    for disease in diseases:
        data.append(lbtestdata[lbtestdata['NECROSIS'] == 1][lbtest].values)


    data = pd.DataFrame(data).T

    min_ = data.min().min()

    data = data.applymap(lambda x: math.log10(x+min_+1))

    data.columns = ['NEGATIVE'] + ['NECROSIS']

    df_long = pd.melt(data)

    colors = [(0, 0, 0, 0.4),
              (255/255, 153/255, 153/255, 1)]

    palette = dict(zip(data.columns, colors))


    ax = sns.boxplot(x="variable", y="value",
                     data=df_long, showfliers=True if not swarm else False,
                     palette=palette, ax=axarr.flatten()[i])

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.3))
    if swarm:
        ax = sns.swarmplot(x="variable", y="value", data=df_long,
                           palette=palette, alpha=0.6,
                           ax=ax, edgecolor='black', linewidth=1.0)

    ax.set_ylabel("log({}/control)".format(lbtest), fontsize=16)

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    ax.set_xlabel("")


plt.savefig(os.path.join(figures_dir, 'plot.png'), transparent=True)