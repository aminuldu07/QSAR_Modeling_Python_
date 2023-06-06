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

df = data.replace(np.inf, np.nan)
df = df.drop(['IS_CONTROL'], axis=1)
df = df.fillna(df.mean())


top10cols = ['ALB-SERUM', 'ALBGLOB-SERUM', 'ALP-SERUM', 'ALT-SERUM', 'APTT-PLASMA',
       'AST-SERUM', 'BASO-WHOLE BLOOD', 'BILI-SERUM', 'CA-SERUM', 'CHOL-SERUM',
       'CK-SERUM', 'CL-SERUM', 'CREAT-SERUM', 'EOS-WHOLE BLOOD',
       'FIBRINO-PLASMA', 'GGT-SERUM', 'GLOBUL-SERUM', 'GLUC-SERUM',
       'HCT-WHOLE BLOOD', 'HGB-WHOLE BLOOD', 'K-SERUM', 'LGUNSCE-WHOLE BLOOD',
       'LYM-WHOLE BLOOD', 'MCH-WHOLE BLOOD', 'MCHC-WHOLE BLOOD',
       'MCV-WHOLE BLOOD', 'MONO-WHOLE BLOOD', 'NEUT-WHOLE BLOOD', 'PH-URINE',
       'PHOS-SERUM', 'PLAT-WHOLE BLOOD', 'PROT-SERUM', 'PROT-URINE',
       'PT-PLASMA', 'RBC-WHOLE BLOOD', 'RDW-WHOLE BLOOD', 'RETI-WHOLE BLOOD',
       'RETIRBC-WHOLE BLOOD', 'SODIUM-SERUM', 'SPGRAV-URINE', 'TRIG-SERUM',
       'UREAN-SERUM', 'VOLUME-URINE', 'WBC-WHOLE BLOOD']

scaler = StandardScaler()

df[top10cols] = scaler.fit_transform(df[top10cols].apply(lambda x: math.log10(x+abs(x.min())+1)))

df = df[top10cols+['USUBJID']].melt(id_vars='USUBJID', value_vars=top10cols).dropna()
df = df.merge(data[['USUBJID', 'SEX']])
print(df)
# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="variable", hue="SEX", aspect=10, height=.5)

# Draw the densities in a few steps
g.map(sns.kdeplot, "value", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "value", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "variable")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

plt.savefig(os.path.join(config.IMG_DIR, 'joyplot.png'), transparent=True)