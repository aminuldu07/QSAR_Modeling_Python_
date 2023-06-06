

from hepatotox_db import send_liver
from send import send_db
import pandas as pd, numpy as np

# what needs to be done is identify studies that only have a
# control, low medium and high group

ts_trts = send_db.generic_query('SELECT APPNUMBER, TS.STUDYID, upper(TSVAL) as TREATMENT FROM TS '
                                'INNER JOIN AN ON TS.STUDYID == AN.STUDYID '
                                'WHERE TSPARMCD == "TRT"')

ex = send_db.generic_query('SELECT APPNUMBER, EX.STUDYID, USUBJID, upper(EXTRT) as TRTMNT FROM EX INNER JOIN '
                           'AN ON EX.STUDYID == AN.STUDYID')


rat_studies = send_liver.generic_query('SELECT DISTINCT STUDYID FROM TERMINALRAT')

appchems = send_liver.generic_query('SELECT ID, APPNUMBER, BDNUMS from APPID')

ex_app = ex.loc[ex.STUDYID.isin(rat_studies.STUDYID)].merge(appchems)


trt_frq = ex_app.groupby('APPNUMBER')['TRTMNT'].nunique()

ones = ex_app[ex_app.APPNUMBER.isin(trt_frq[trt_frq == 1].index)]



# for gp, data in ones.groupby('APPNUMBER'):
#     print(data.TRTMNT.unique())
#
# twos = ex_app[ex_app.APPNUMBER.isin(trt_frq[trt_frq > 1].index)]
# twos = twos.drop_duplicates(['USUBJID', 'TRTMNT'])
#
# for gp, data in twos.groupby('APPNUMBER'):
#     print(data.STUDYID.unique(), data.TRTMNT.value_counts())