# find the matched sdf files by matching "sdf files" & a csv files containning "smile string"
# "sdf files" & "smile string" matching

import os
import shutil
import pandas as pd
from rdkit import Chem

# Define the directory where SDF files are located
sdf_directory = 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/cleaned_sdf_single'

# Define the directory where to put the matching SDF files
output_directory = 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/sdftosmile'

# Load the CSV file into a pandas dataframe
df = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/df/U_SMILES.csv')

# Convert the SMILES column into a set for faster lookup
smiles_set = set(df['SMILES'])

# Define the function to extract the SMILES string from an SDF file
def get_smiles_from_sdf(file_path):
    supplier = Chem.SDMolSupplier(file_path)
    for mol in supplier:
        if mol is not None:
            return Chem.MolToSmiles(mol)
    return None

# Loop over all SDF files in the directory
for file_name in os.listdir(sdf_directory):
    if file_name.endswith('.sdf'):
        file_path = os.path.join(sdf_directory, file_name)
        smiles = get_smiles_from_sdf(file_path)
        if smiles in smiles_set:
            # If a match is found, copy the file to the output directory
            shutil.copy(file_path, output_directory)

# "Exception error handling" for the above function 
def get_smiles_from_sdf(file_path):
    try:
        supplier = Chem.SDMolSupplier(file_path)
        for mol in supplier:
            if mol is not None:
                return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return None

# Loop over all SDF files in the directory
for file_name in os.listdir(sdf_directory):
    if file_name.endswith('.sdf'):
        file_path = os.path.join(sdf_directory, file_name)
        smiles = get_smiles_from_sdf(file_path)
        if smiles in smiles_set:
            # If a match is found, copy the file to the output directory
            shutil.copy(file_path, output_directory)

            
            
            
            
