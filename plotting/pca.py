from hepatotox_db import send_liver
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

features = send_liver.load_chemical_features('rdkit')

features = minmax_scale(features)

pca = PCA(2).fit_transform(features)

fig, ax = plt.subplots()

ax.scatter(pca[:, 0], pca[:, 1])

plt.show()