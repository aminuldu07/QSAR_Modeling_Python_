import numpy as np, pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import math
from hepatotox_db import send_liver
import config, os

data = send_liver.generic_query('SELECT * FROM TERMINALRAT')


features = ['ALT', 'AST', 'BILI', 'ALP']
classes = ['CHOLESTASIS', 'STEATOSIS', 'NECROSIS']

data = data[features + classes]
data = data[data.notnull().all(1)]

X = (data[features] + 1).applymap(lambda x: math.log(x))


for i in range(len(classes)):
    disease = classes[i]
    y = data[classes]

    y = y[disease]
    fig, axarr = plt.subplots(len(features), len(features), sharey='row', sharex='col')

    disease_colors = dict(zip(['CHOLESTASIS', 'STEATOSIS', 'NECROSIS'], [(0, 1, 0, 0.6), (1, 0.647, 0, 0.6), (1, 0, 0, 0.6)]))


    colors = [(0, 0, 0, 0.2), disease_colors[disease]]
    size = 25
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
                axarr[row, col].scatter(X[test_j][y == 0], X[test_i][y == 0],
                                        s=size,
                                        c=colors[0], zorder=0, edgecolor='k', lw=lw)


                axarr[row, col].scatter(X[test_j][y == 1], X[test_i][y == 1],
                                        s=size,
                                        c=colors[1], zorder=1, edgecolor='k', lw=lw)
                print(col, row)
                if col == 0:
                    axarr[row, col].set_ylabel(test_i)
                    axarr[row, col].set_yticks(np.arange(0, 4))
                if row == len(features)-1:
                    axarr[row, col].set_xlabel(test_j)
                    axarr[row, col].set_xticks(np.arange(0, 4))


    plt.suptitle(disease.title(), fontsize=18)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(os.path.join(config.IMG_DIR, 'labtest_{}.png'.format(disease)), transparent=True)
