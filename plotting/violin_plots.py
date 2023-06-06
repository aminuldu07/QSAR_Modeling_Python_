import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config, os, math

from sklearn.preprocessing import StandardScaler

sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create the data
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list("ABCDEFGHIJ"), 50)
data = pd.read_csv('../data_scratch/new_rat_data.csv', index_col=0).drop(['STUDYID', 'NECROSIS',
                                                                        'SPECIES', 'STEATOSIS', 'CHOLESTASIS',
                                                                        'BWDIFF_NORM', 'BWDIFF', 'BWINTCEPT_NORM',
                                                                        'BWINTCEPT', 'BWSLOPE', 'BWSLOPE_NORM'], axis=1)

df = data[data.IS_CONTROL]
df = df.drop(['IS_CONTROL'], axis=1)
good_cols = [col for col in df.columns if 'SERUM' in col or 'WHOLE BLOOD' in col]

top10cols = df[good_cols].notnull().sum().sort_values(ascending=False).index[:20].tolist()
# top10cols = top10cols[:5].tolist() + top10cols[-5:].tolist()
# top10cols = top10cols[:20]

scaler = StandardScaler()

# df[top10cols] = scaler.fit_transform(df[top10cols].applymap(lambda x: math.log10(x+1)))
df[top10cols] = df[top10cols].applymap(lambda x: math.log10(x+1))

df = df[top10cols+['USUBJID']].melt(id_vars='USUBJID', value_vars=top10cols).dropna()
df = df.merge(data[['USUBJID', 'SEX']])

sns.set(style="whitegrid", palette="pastel", color_codes=True)

plt.figure(figsize=plt.figaspect(1/1.5))
# Draw a nested violinplot and split the violins for easier comparison
ax = sns.violinplot(x="variable", y="value", hue="SEX",
               split=True,
               inner="quart",
               # palette={"Male": "M", "Female": "F"},
               data=df)
sns.despine(left=True)

# ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax.set_xlabel('')
ax.set_ylabel('Log(Test/Mean(Control))')
# ax.set_yticklabels(['', 'Control Mean', '', '', '', ''])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(config.IMG_DIR, 'violintplot.png'), transparent=True)