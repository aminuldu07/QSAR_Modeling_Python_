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


diseases = ['STEATOSIS', 'CHOLESTASIS', 'NECROSIS']

lbtests = ['ALT', 'AST', 'BILI', 'ALP']

df = rats.db[rats.db.SEX == sex][lbtests + diseases]

fig, axarr = plt.subplots(ncols=2,
                          nrows=2,
                          sharex=True,
                          figsize=plt.figaspect(1/1.5)*2)

swarm = True

for i, lbtest in enumerate(lbtests):

    data = []
    lbtestdata = df[df[lbtest].notnull()]

    data.append(lbtestdata[lbtestdata[diseases].sum(1) == 0][lbtest].values)

    for disease in diseases:
        data.append(lbtestdata[lbtestdata[disease] == 1][lbtest].values)


    data = pd.DataFrame(data).T

    min_ = data.min().min()

    data = data.applymap(lambda x: math.log10(x+min_+1))

    data.columns = ['No\nObservable\nDisease'] + [d.title() for d in diseases]

    df_long = pd.melt(data)

    colors = [(0, 0, 0, 0.4),
              (255/255, 153/255, 153/255, 1),
              (153/255, 204/255, 153/255, 1),
              (153/255, 153/255, 255/255, 1)]

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


plt.savefig(os.path.join(figures_dir, 'Figure3_{}_{}.png'.format(sex, 'swarm' if swarm else '')), transparent=True)