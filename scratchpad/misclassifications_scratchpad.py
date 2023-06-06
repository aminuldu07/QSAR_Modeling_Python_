from send import send_db
from hepatotox_db import send_liver
import seaborn as sns
import matplotlib.pyplot as plt

predictions = send_liver.generic_query('SELECT * from PREDICTIONS')
params = send_liver.generic_query('SELECT * from PARAMS')
predictions = predictions.merge(params)
data = send_liver.generic_query('SELECT * FROM TERMINALRAT')

mdl = 'rf'
cv = '10'
features = 'RBC,ALBGLOB,BWSLOPE_NORM,CK,K,TRIG,BWINTCEPT_NORM,PROT,PHOS,UREAN,MCV,MCHC,GGT,RETIRBC,BASO,CHOL,LYM,BILI,NEUTLE,PLAT,CREAT,EOSLE,ALP,RETI,LYMLE,HGB,AST,SEX,MONOLE,ALT,BASOLE,BWDIFF_NORM,EOS,SODIUM,WBC,NEUT,LGUNSCE,HCT,CA,ALB,MONO,GLOBUL,RDW,CL,MCH,GLUC'
percent = 0.55
disease = 'NECROSIS'

rf_feature_rankings = {
    'NECROSIS': ['GGT', 'BILI', 'CL', 'RBC', 'MCHC', 'PROT', 'BWSLOPE_NORM', 'GLOBUL', 'SEX', 'ALP',
                 'NEUTLE', 'CHOL', 'EOSLE', 'BASO', 'BASOLE', 'GLUC', 'MCH', 'PHOS', 'LYM', 'CK',
                 'RDW', 'RETI', 'CREAT', 'TRIG', 'SODIUM', 'EOS', 'MCV', 'BWINTCEPT_NORM', 'HGB',
                 'LGUNSCE', 'WBC', 'K', 'MONOLE', 'HCT', 'RETIRBC', 'BWDIFF_NORM', 'UREAN', 'PLAT',
                 'CA', 'MONO', 'ALT', 'LYMLE', 'NEUT', 'ALBGLOB', 'AST', 'ALB'],
    'STEATOSIS': ['SEX', 'PHOS', 'PLAT', 'GGT', 'BASOLE', 'ALBGLOB', 'K', 'RDW', 'SODIUM', 'GLOBUL',
                  'EOSLE', 'TRIG', 'MCHC', 'CL', 'NEUTLE', 'BWINTCEPT_NORM', 'CHOL', 'MONOLE',
                  'BASO', 'BILI', 'MONO', 'RETI', 'WBC', 'CK', 'MCV', 'GLUC', 'LYMLE', 'MCH', 'UREAN',
                  'LYM', 'ALT', 'BWDIFF_NORM', 'EOS', 'RETIRBC', 'ALP', 'PROT', 'AST', 'CREAT',
                  'HCT', 'ALB', 'LGUNSCE', 'NEUT', 'HGB', 'BWSLOPE_NORM', 'CA', 'RBC'],
    'CHOLESTASIS': ['SEX', 'CA', 'LGUNSCE', 'MCV', 'HGB', 'MCH', 'LYM', 'WBC', 'LYMLE', 'RETIRBC',
                    'AST', 'RETI', 'MONO', 'BWINTCEPT_NORM', 'EOSLE', 'MONOLE', 'K', 'PHOS', 'GLUC',
                    'EOS', 'RBC', 'HCT', 'NEUT', 'BASO', 'NEUTLE', 'RDW', 'MCHC', 'TRIG', 'CHOL', 'ALB',
                    'BASOLE', 'BWDIFF_NORM', 'ALBGLOB', 'BWSLOPE_NORM', 'PLAT', 'PROT', 'SODIUM',
                    'UREAN', 'GGT', 'BILI', 'CK', 'ALP', 'GLOBUL', 'CL', 'CREAT', 'ALT']


}


new_predictions = predictions.copy()
new_predictions = new_predictions[((new_predictions.MODEL_TYPE == mdl) &
                                   (new_predictions.CV == cv) &
                                   (new_predictions.FEATURES == features) &
                                   (new_predictions.MINOR_CLASS_PERCENT == percent))]

new_predictions['AVG'] = new_predictions.groupby(['DISEASE', 'USUBJID'])['PREDICTION'].transform('mean')
diseases = new_predictions.DISEASE.unique()
new_data = data[['USUBJID', 'SEX'] + diseases.tolist()].copy()
new_predictions = new_data.merge(new_predictions)

disease_predictions = new_predictions[new_predictions.DISEASE == disease]
disease_predictions = disease_predictions[['USUBJID', 'AVG', disease]].drop_duplicates()

disease_predictions = disease_predictions[disease_predictions.AVG.notnull()]

d = disease_predictions.merge(data[['USUBJID'] + features.split(',')])

d = d[features.replace('SEX,', '').split(',')]
import math
# d = d.applymap(lambda x: x+1 if x >= 0 else 1)
#
# d = d.applymap(lambda x: math.log10(x))

d = d[rf_feature_rankings[disease]]

ax = sns.clustermap(d, col_cluster=True, row_cluster=False, z_score=1)

plt.show()