from send import send_db
import datetime
import dateutil.parser
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import config, os

study_start = send_db.generic_query("SELECT STUDYID, TSPARMCD, TSVAL FROM TS where TSPARMCD='STSTDTC'")

animals = send_db.generic_query("SELECT STUDYID, USUBJID from DM")
print(animals.shape)
print(study_start['TSVAL'])

def fun(x):
    try:
        return dateutil.parser.parse(x)
    except:
        print(x)
        return np.nan

study_start['datetime'] = study_start['TSVAL'].apply(fun)
study_start = study_start[study_start['datetime'].notnull()]


data = pd.merge(study_start, animals)
data.index = data.datetime

grouped_by_month = data.groupby(pd.Grouper(freq="Y"))

new_frame = pd.DataFrame(columns=['M', 'Y', ])

list_data = []

for month, d in grouped_by_month:

    month_submissions = d.STUDYID.value_counts().count()
    month_animals = d.USUBJID.value_counts().count()
    list_data.append((month.month, month.year, month_submissions, month_animals))

df = pd.DataFrame(list_data, columns=['M', 'Y', 'STUDIES', 'ANIMALS'])

df = df.sort_values(['M', 'Y'])

df['total_studies'] = df['STUDIES'].cumsum()
df['total_animals'] = df['ANIMALS'].cumsum()

df = df[df.Y > 2005]

idx = np.arange(len(df))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(1/2))


ax[0].bar(idx, df['STUDIES'], facecolor=(0, 0, 1, 0.4), edgecolor='k', lw=2)
ax[0].plot(idx, df['total_studies'], color=(0, 0, 1, 0.4), lw=2)
ax[0].scatter(idx, df['total_studies'], facecolor=(0, 0, 1, 0.6), edgecolor='k')

ax[1].bar(idx, df['ANIMALS'], facecolor=(1, 0, 0, 0.4), edgecolor='k', lw=2)
ax[1].plot(idx, df['total_animals'], color=(1, 0, 0, 0.4), lw=2)
ax[1].scatter(idx, df['total_animals'], facecolor=(1, 0, 0, 0.6), edgecolor='k')

for a in ax.flatten():
    a.set_xticks(idx)
    a.set_xticklabels(['{y}'.format(y=y) for m, y in zip(df['M'], df['Y'])], rotation=45)
    a.set_xlabel('Study Start Date')

ax[1].set_yticks([1600] + list(ax[1].get_yticks())[1:])
ax[1].get_yticklabels()[0].set_color((0, 0, 1, 0.6))
ax[1].get_yticklabels()[0].set_fontweight("bold")

ax[0].set_title('Studies')
ax[1].set_title('Animals')
ax[0].set_ylabel('Count')

_, y_max = ax[0].get_ylim()

ax[0].set_ylim([0, y_max])


plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(os.path.join(config.IMG_DIR, 'studies_animals_year.png'), transparent=True)