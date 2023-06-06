from hepatotox_db import send_liver

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

import pandas as pd

rats_chemicals = send_liver.load_chemicals(animals=True)

mols = []

for i, data in rats_chemicals.iterrows():
    mol = Chem.MolFromMolBlock(data['MOL_BLOCK'])
    mol.SetProp('ID', data['ID'])
    mols.append(mol)


rats_chemicals['rdkit'] = mols


chemicals = rats_chemicals[['ID', 'rdkit']].drop_duplicates('ID')



def calc_rdkit(molecules, name_col='ID'):
    """
    Takes in a list of rdkit molecules, calculates molecular descriptors for each molecule, and returns a machine
    learning-ready pandas DataFrame.
    :param molecules: List of rdkit molecule objects with no None values
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    # Generates molecular descriptor calculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])

    # Calculates descriptors and stores in pandas DataFrame
    X = pd.DataFrame([list(calculator.CalcDescriptors(mol)) for mol in molecules],
                     index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules],
                     columns=list(calculator.GetDescriptorNames()))

    # Imputes the data and replaces NaN values with mean from the column
    desc_matrix = X.copy()

    # Checks for appropriate output
    assert len(desc_matrix.columns) != 0, 'All features contained at least one null value. No descriptor matrix ' \
                                          'could be generated.'

    return desc_matrix


def calc_ecfp6(molecules, name_col='ID', use_chirality=True):
    """
    Takes in a list of rdkit molecules and returns ECFP6 fingerprints for a list of rdkit molecules
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecule objects with no None values
    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    data = []

    for mol in molecules:
        ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024, useChirality=use_chirality)]
        data.append(ecfp6)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def calc_fcfp6(molecules, name_col='ID', use_chirality=True):
    """
    Takes in a list of rdkit molecules and returns FCFP6 fingerprints for a list of rdkit molecules
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecules with no None values
    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    data = []

    for mol in molecules:
        fcfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024, useFeatures=True, useChirality=use_chirality)]
        data.append(fcfp6)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])


def calc_maccs(molecules, name_col='ID'):
    """
    Takes in a list of rdkit molecules and returns MACCS fingerprints for a list of rdkit molecules
    :param name_col: Name of the field to index the resulting DataFrame.  Needs to be a valid property of all molecules
    :param molecules: List of rdkit molecules with no None values
    :return: pandas DataFrame of dimensions m x n, where m = # of descriptors and n = # of molecules
    """

    # Checks for appropriate input
    assert isinstance(molecules, list), 'The molecules entered are not in the form of a list.'
    assert all((isinstance(mol, Chem.rdchem.Mol) for mol in molecules)), 'The molecules entered are not rdkit Mol ' \
                                                                         'objects.'
    assert None not in molecules, 'The list of molecules entered contains None values.'
    assert isinstance(name_col, str), 'The input parameter name_col (%s) must be a string.' % name_col

    data = []

    for mol in molecules:
        maccs = [int(x) for x in MACCSkeys.GenMACCSKeys(mol)]
        data.append(maccs)

    return pd.DataFrame(data, index=[mol.GetProp(name_col) if mol.HasProp(name_col) else '' for mol in molecules])



descriptor_fxs = {
        'rdkit': lambda mols: calc_rdkit(mols),
        'ECFP6': lambda mols: calc_ecfp6(mols),
        'FCFP6': lambda mols: calc_fcfp6(mols),
        'MACCS': lambda mols: calc_maccs(mols)
    }

conn = send_liver.connect_to_liverdb()

for desc, fx in descriptor_fxs.items():
    df = fx(chemicals['rdkit'].values.tolist())
    if desc == 'rdkit':
        df = df.reset_index().rename({'index': 'ID'}, axis=1)
        df.to_sql('{}'.format(desc), con=conn, if_exists='replace', index=False)
    else:
        df = df.apply(lambda row: ','.join(row[row == 1].index.astype(str).tolist()), axis=1)
        df = df.reset_index().rename({'index': 'ID', 0: 'on_bits'}, axis=1)
        df.to_sql('{}'.format(desc), con=conn, if_exists='replace', index=False)

conn.close()
