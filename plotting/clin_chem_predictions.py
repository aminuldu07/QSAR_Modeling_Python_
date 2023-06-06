import numpy as np, pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import math
from hepatotox_db import send_liver
import config, os

data = send_liver.generic_query('SELECT * FROM TERMINALRAT')


features = ['ALT', 'AST', 'BILI', 'ALP']
classes = ['CHOLESTASIS', 'STEATOSIS', 'NECROSIS']

data = data[features + classes]
data = data[data.notnull().all(1)]

X = (data[features] + 1).applymap(lambda x: math.log(x))

predictions = pd.read_csv(os.path.join(config.TEXT_DIR, 'rf_predictions.csv'), index_col=0)


for i in range(len(classes)):
    disease = classes[i]

    classification = data[disease].values
    y = predictions[disease]
    fig, axarr = plt.subplots(len(features), len(features), sharey='row', sharex='col')

    disease_colors = dict(zip(['CHOLESTASIS', 'STEATOSIS', 'NECROSIS'], [(0, 1, 0, 0.6), (1, 0.647, 0, 0.6), (1, 0, 0, 0.6)]))

    r, g, b, a = disease_colors[disease]

    cmap = colors.LinearSegmentedColormap.from_list('mycmap', [(0, 0, 0, 0), (r, g, b, 1)])


    size = 10
    lw=0.5

    for row, test_i in enumerate(features):
        for col, test_j in enumerate(features):
            if row == col:
                # axarr[row, col].set_facecolor((0, 0, 0, 0.2))
                fig.patch.set_facecolor('blue')
                fig.patch.set_alpha(0.7)
                if row == len(features)-1:

                    axarr[row, col].set_xlabel(test_j)
                    axarr[row, col].set_xticks(np.arange(0, 4))
                if col == 0:
                    axarr[row, col].set_ylabel(test_i)
                    axarr[row, col].set_yticks(np.arange(0, 4))
            else:
                axarr[row, col].scatter(X[test_j][classification == 1], X[test_i][classification == 1],
                                        s=size+10,
                                        c=y[classification == 1],
                                        cmap=cmap,
                                        marker='^',
                                        edgecolor='k', lw=lw, zorder=1)
                axarr[row, col].scatter(X[test_j][classification == 0], X[test_i][classification == 0],
                                        s=size,
                                        c=y[classification == 0],
                                        cmap=cmap,
                                        marker='o',
                                        edgecolor='k', lw=lw, zorder=0)


                print(col, row)
                if col == 0:
                    axarr[row, col].set_ylabel(test_i)
                    axarr[row, col].set_yticks(np.arange(0, 4))
                if row == len(features)-1:
                    axarr[row, col].set_xlabel(test_j)
                    axarr[row, col].set_xticks(np.arange(0, 4))

    cbar_ax = fig.add_axes([0.9, 0.25, 0.025, 0.5])

    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, shrink=0.6)
    plt.suptitle(disease.title(), fontsize=18)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(os.path.join(config.IMG_DIR, 'labtest_predictions_{}.png'.format(disease)), transparent=True)
