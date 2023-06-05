
import os
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def process_sdf_files(sdf_directory, output_directory, csv_path):
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(csv_path)

    # Convert the SMILES column into a set for faster lookup
    smiles_set = set(df['SMILES'])

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

    # Get a list of all descriptor names
    descriptor_names = [x[0] for x in Descriptors._descList]

    # Initialize an empty DataFrame to store the descriptors
    descriptors_df = pd.DataFrame(columns=['Filename'] + descriptor_names)

    # Loop over each SDF file in the output directory
    for sdf_file in os.listdir(output_directory):
        if sdf_file.endswith('.sdf'):
            # Full path to the SDF file
            sdf_path = os.path.join(output_directory, sdf_file)

            # Read the SDF file
            supplier = Chem.SDMolSupplier(sdf_path)

            # Loop over each molecule in the SDF file
            for mol in supplier:

                if mol is None:
                    continue

                # Initialize a dictionary to store the descriptors for this molecule
                descriptors_dict = {'Filename': sdf_file}

                # Calculate all descriptors
                for descriptor_name in descriptor_names:

                    # Get the descriptor function
                    descriptor_func = Descriptors.__dict__[descriptor_name]

                    # Calculate the descriptor
                    descriptor_value = descriptor_func(mol)

                    # Add the descriptor to the dictionary
                    descriptors_dict[descriptor_name] = descriptor_value

                # Append the descriptors for this molecule to the DataFrame
                descriptors_df = pd.concat([descriptors_df, pd.DataFrame(descriptors_dict, index=[0])], ignore_index=True)
                
    return descriptors_df
### You can then call this function like so:
sdf_directory = 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/cleaned_sdf_single'
output_directory = 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/sdftosmile'
csv_path = 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/df/U_SMILES.csv'
df = process_sdf_files(sdf_directory, output_directory, csv_path)
print(df)
