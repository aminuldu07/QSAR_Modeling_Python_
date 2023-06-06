import numpy as np
from hepatotox_db import send_liver
import pickle
import matplotlib.pyplot as plt

data = send_liver.generic_query('SELECT * FROM TERMINALRAT')
data.index = data.USUBJID

necrosis_model = send_liver.generic_query('SELECT MODEL, MODELS.ID, METRIC, VALUE from MODELS ' 
                                          'INNER JOIN PARAMS ON MODELS.id = PARAMS.id '
                                          'INNER JOIN STATS ON PARAMS.id = STATS.id '
                                          'WHERE PARAMS.DISEASE = "NECROSIS"')

best_model = necrosis_model[(necrosis_model.METRIC == 'AUC')]
best_model = necrosis_model[(necrosis_model.VALUE == necrosis_model.VALUE.max())]
best_model = pickle.loads(best_model.model.iloc[0])

features = ['ALT', 'AST', 'BILI', 'ALP']

data = data[features]
data = data[data.notnull().all(1)]

probas = best_model.predict_proba(data)[:, 1]



def plot_edish(ax, data, x_col, y_col, probas):

    xx, yy = np.mgrid[data[x_col].min()-10:data[x_col].max()+10:1, data[y_col].min()-10:data[y_col].max()+10:1]

    fake_data_list = [None] * data.shape[1]

    x_idx = data.columns.get_loc(x_col)
    y_idx = data.columns.get_loc(y_col)

    for i, col in enumerate(data.columns):
        if i == x_idx:
            fake_data_list[i] = xx.ravel()
        elif i == y_idx:
            fake_data_list[i] = yy.ravel()
        else:
            fake_data_list[i] = np.full(xx.ravel().shape, data.iloc[:, i].mean())

    fake_data = np.c_[tuple(fake_data_list)]

    probs = best_model.predict_proba(fake_data)[:, 1].reshape(xx.shape)


    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu_r",
                          vmin=probs.min(), vmax=probs.max())
    # ax_c = fig.colorbar(contour)
    # ax_c.set_label("$P(y = 1)$")
    # ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(data[x_col], data[y_col], c=probas, s=50,
               cmap="RdBu_r", vmin=0, vmax=1,
               edgecolor="white", linewidth=1)


fig, axarr = plt.subplots(len(features), len(features))

for row, test_i in enumerate(features):
    for col, test_j in enumerate(features):
        if row == col:
            axarr[row, col].set_facecolor((0, 0, 0, 0.2))
            # fig.patch.set_facecolor('blue')
            # fig.patch.set_alpha(0.7)
            if row == len(features) - 1:
                axarr[row, col].set_xlabel(test_j)
            if col == 0:
                axarr[row, col].set_ylabel(test_i)
        else:
            axarr[row, col] = plot_edish(axarr[row, col], data, test_j, test_i, probas)

plt.show()