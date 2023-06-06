import pickle
from hepatotox_db import send_liver
import numpy as np


stats = send_liver.generic_query('select id, value from stats where metric == "AUC"')

params = send_liver.generic_query('select * from params')

# models = models.merge(stats).merge(params)

terminal_rats = send_liver.generic_query('SELECT * FROM TERMINALRAT')


mdl = 'rf'
cv = '10'
features = 'RBC,ALBGLOB,BWSLOPE_NORM,CK,K,TRIG,BWINTCEPT_NORM,PROT,PHOS,UREAN,MCV,MCHC,GGT,RETIRBC,BASO,CHOL,LYM,BILI,NEUTLE,PLAT,CREAT,EOSLE,ALP,RETI,LYMLE,HGB,AST,SEX,MONOLE,ALT,BASOLE,BWDIFF_NORM,EOS,SODIUM,WBC,NEUT,LGUNSCE,HCT,CA,ALB,MONO,GLOBUL,RDW,CL,MCH,GLUC'
percent = 0.55

for disease in ['NECROSIS', 'CHOLESTASIS', 'STEATOSIS']:
    new_params = params.copy()
    new_params = new_params[((new_params.MODEL_TYPE == mdl) &
                                       (new_params.CV == cv) &
                                       (new_params.FEATURES == 'RBC,ALBGLOB,BWSLOPE_NORM,CK,K,TRIG,BWINTCEPT_NORM,PROT,PHOS,UREAN,MCV,MCHC,GGT,RETIRBC,BASO,CHOL,LYM,BILI,NEUTLE,PLAT,CREAT,EOSLE,ALP,RETI,LYMLE,HGB,AST,SEX,MONOLE,ALT,BASOLE,BWDIFF_NORM,EOS,SODIUM,WBC,NEUT,LGUNSCE,HCT,CA,ALB,MONO,GLOBUL,RDW,CL,MCH,GLUC') &
                                       (new_params.MINOR_CLASS_PERCENT == percent) &
                                        (new_params.DISEASE == disease))]

    models = send_liver.generic_query('select * from models')
    best_models = new_params.merge(models)

    models = []
    for model in best_models.MODEL:

        models.append(pickle.loads(model))


    features_a = np.asarray(features.split(','))

    importances = []

    for m in models:
        importances.append(m.feature_importances_)

    importances = np.asarray(importances)
    print(features_a[np.argsort(importances.mean(0))])