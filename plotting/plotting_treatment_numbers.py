from send import send_db
from hepatotox_db import send_liver
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import os

app_studies = send_db.generic_query('SELECT * FROM AN')

ex = send_db.generic_query('SELECT STUDYID, USUBJID, upper(EXTRT) as TRTMNT, EXDOSE FROM EX')

inds = app_studies[app_studies.APPNUMBER.apply(lambda x: x.startswith('IND'))]

ex_app = ex.merge(inds, on='STUDYID')

trt_frq = ex_app.groupby('APPNUMBER')['TRTMNT'].nunique()
bins = np.arange(0.5, trt_frq.max()+2.5)

fig, ax = plt.subplots()
ax.hist(trt_frq, bins, edgecolor=(0, 0, 0, 1), alpha=0.8)
ax.set_xticks(np.arange(0, trt_frq.max()+2))
ax.set_xlabel('No. of unique terms in EXTRT')
ax.set_title('Frequency of EXTRT terms in {} IND applications'.format(trt_frq.shape[0]))
plt.savefig(os.path.join(cfg.IMG_DIR, 'trt', 'all_studies_freq.png'))

rats = send_liver.generic_query('SELECT * FROM TERMINALRAT')

ex_app = ex_app[ex_app.STUDYID.isin(rats.STUDYID)]

trt_frq = ex_app.groupby('APPNUMBER')['TRTMNT'].nunique()
bins = np.arange(0.5, trt_frq.max()+2.5)

fig, ax = plt.subplots()
ax.hist(trt_frq, bins, edgecolor=(0, 0, 0, 1), alpha=0.8)
ax.set_xticks(np.arange(0, trt_frq.max()+2))
ax.set_xlabel('No. of unique terms in EXTRT')
ax.set_title('Frequency of EXTRT terms in {} IND applications'.format(trt_frq.shape[0]))
plt.savefig(os.path.join(cfg.IMG_DIR, 'trt', 'rat_studies_freq.png'))

chemicals = send_liver.generic_query('SELECT * FROM APPID')

ex_app = ex_app[ex_app.APPNUMBER.isin(chemicals.APPNUMBER)]

trt_frq = ex_app.groupby('APPNUMBER')['TRTMNT'].nunique()
bins = np.arange(0.5, trt_frq.max()+2.5)

fig, ax = plt.subplots()
ax.hist(trt_frq, bins, edgecolor=(0, 0, 0, 1), alpha=0.8)
ax.set_xticks(np.arange(0, trt_frq.max()+2))
ax.set_xlabel('No. of unique terms in EXTRT')
ax.set_title('Frequency of EXTRT terms in {} IND applications'.format(trt_frq.shape[0]))
plt.savefig(os.path.join(cfg.IMG_DIR, 'trt', 'chemicals_rat_studies_freq.png'))

