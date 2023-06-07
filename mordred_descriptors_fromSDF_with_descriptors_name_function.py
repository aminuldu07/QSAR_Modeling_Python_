# ## ## Calculate descriptors using mordred from sdf file 

from mordred import Calculator, descriptors
from rdkit import Chem
import os
import pandas as pd

def calculate_mordred_descriptors(sdf_directory):
    # Initialize a calculator with all available descriptors
    calc = Calculator(descriptors, ignore_3D=True)

    # Get a list of all SDF files in the directory
    sdf_files = [f for f in os.listdir(sdf_directory) if f.endswith('.sdf')]

    # A dictionary to hold all the dataframes
    dict_dfs = {}

    # Loop over each SDF file
    for sdf_file in sdf_files:

        # Full path to the SDF file
        sdf_path = os.path.join(sdf_directory, sdf_file)

        # Read the SDF file
        supplier = Chem.SDMolSupplier(sdf_path)

        # Calculate the descriptors for each molecule
        descriptor_values = [calc(mol) for mol in supplier if mol is not None]

        # Convert the descriptor values to a DataFrame
        df = pd.DataFrame(descriptor_values, columns=descriptor_names)
        df['Filename'] = sdf_file  # adding filename column to dataframe

        # Add the dataframe to the list
        dict_dfs[sdf_file] = df

    # Concatenate all the dataframes
    final_df = pd.concat(dict_dfs, ignore_index=False)
    final_df.reset_index(inplace=True)  # reset index to have level_0 as 'Filename'
    final_df.rename(columns={'level_0': 'Filename'}, inplace=True)  # rename column to 'Filename'
    final_df.set_index('Filename', inplace=True)  # set 'Filename' as index
    
    return final_df
#write the csv file
final_df = calculate_mordred_descriptors('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/sdftosmile')
