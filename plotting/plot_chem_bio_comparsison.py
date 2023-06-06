from models_db import MODEL
from hepatotox_db import ANIMALDB
import pandas as pd

species = 'RAT'
fold = 10
ratio = 0.55
bio_features = 'ALB,ALBGLOB,ALP,ALT,AST,BASO,BILI,CA,CHOL,CL,CREAT,EOS,GLOBUL,GLUC,HCT,HGB,K,LGUNSCE,LYM,MCH,MCHC,MCV,MONO,NEUT,PHOS,PLAT,PROT,RBC,RETI,SODIUM,TRIG,WBC,SEX,BWDIFF_NORM,BWSLOPE_NORM,BWINTCEPT_NORM'
diseases = ['NECROSIS', 'STEATOSIS', 'CHOLESTASIS']
chem_features = ['maccs', 'ecfp6', 'fcfp6']

for disease in diseases:
    print(disease)
    y = ANIMALDB(species).get_liver_disease(disease)
    bio = MODEL(species, disease, fold, 'rf', ratio, bio_features, 'None')
    results = []
    df = bio.get_stats().drop('ID', axis=1).set_index('METRIC').rename({'VALUE': 'BIO'}, axis=1)
    results.append(df)
    print(bio.get_preds().shape)
    print((y.loc[bio.get_preds().USUBJID]==1).sum())

    for chem_feature in chem_features:
        print(chem_feature)
        chem = MODEL(species, disease, fold, 'rf', ratio, bio_features, chem_feature)
        df = chem.get_stats().drop('ID', axis=1).set_index('METRIC').rename({'VALUE': chem_feature}, axis=1)
        results.append(df)
        print(chem.get_preds().shape)
        print((y.loc[chem.get_preds().USUBJID] == 1).sum())



    final_df = pd.concat(results, axis=1)
    final_df.to_csv('../data_scratch/{}.csv'.format(disease))

preds = {}

for disease in diseases:
    print(disease)
    bio = MODEL(species, disease, fold, 'rf', ratio, bio_features, 'None')

    df = bio.get_preds().drop('ID', axis=1).set_index('USUBJID').rename({'PREDICTION': 'BIO'}, axis=1)
    predictions = []
    predictions.append(df)
    for chem_feature in chem_features:
        chem = MODEL(species, disease, fold, 'rf', ratio, bio_features, chem_feature)
        df = chem.get_preds().drop('ID', axis=1).set_index('USUBJID').rename({'PREDICTION': chem_feature}, axis=1)
        predictions.append(df)

    predictions[0] = predictions[0].loc[df.index]
    final_df = pd.concat(predictions, axis=1)
    final_df.to_csv('../data_scratch/{}_predictions.csv'.format(disease))
    preds[disease] = final_df
