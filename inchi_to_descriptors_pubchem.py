# code chunk that takes the InChIKeys from the column "InChIKey" in the input CSV file, retrieves the
# corresponding InChI strings from PubChem, calculates the RDKit descriptors for each molecule, and saves
# the descriptors to a new CSV file:

import pubchempy as pcp
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd

# Read the input CSV file with InChIKeys
df = pd.read_csv('INCHIKEY_to_inchi_rdeleted_543r.csv')

# Create an empty list to store the calculated descriptors
descriptor_list = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    inchi_key = row['InChIKey']

    # Get the corresponding compound from PubChem
    compound = pcp.Compound.from_cid(pcp.get_cids(inchi_key, 'inchikey')[0])

    # Get the InChI from the compound
    inchi = compound.inchi

    # Convert the InChI string to a Mol object
    molecule = Chem.MolFromInchi(inchi)

    # Get the list of all available descriptors in RDKit
    descriptors_list = [x[0] for x in Descriptors._descList]

    # Create a molecule descriptor calculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_list)

    # Calculate the descriptors for the molecule
    descriptors = calculator.CalcDescriptors(molecule)

    # Append the descriptors to the list
    descriptor_list.append(descriptors)

# Create a DataFrame for the descriptors
df_descriptors = pd.DataFrame(descriptor_list, columns=descriptors_list)

# Concatenate the original DataFrame and the descriptors DataFrame
df_final = pd.concat([df, df_descriptors], axis=1)

# Save the final DataFrame with descriptors to a new CSV file
df_final.to_csv('descriptors.csv', index=False)
