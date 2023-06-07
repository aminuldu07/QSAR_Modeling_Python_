# Calculate the mordred descriptors from "inchi string"

# Convert the descriptors to a dictionary
# Handle non-serializable descriptors by converting them to a string representation


import os
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import json

# Get the current directory of the Python script
#current_dir = os.path.dirname(os.path.abspath(__file__))
# In Python, __file__ is a special variable provided by the interpreter that indicates the
#path of the currently executing script file. This variable is not defined in the interactive
# Python interpreter or when running a Python script in a Jupyter notebook because these environments
# do not run a script file.

# Get the current directory of the notebook script
import os
current_dir = os.getcwd()
#current_dir


# Read the input CSV file
path = 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/INCHIKEY_to_inchi_rdeleted_543r.csv'
df = pd.read_csv(path)
#df

# File paths
input_csv_path = os.path.join(current_dir, 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/INCHIKEY_to_inchi_rdeleted_543r.csv')
output_csv_path = os.path.join(current_dir, 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/inchi_to_descriptors_mordred_2.csv')

# Read the input CSV file
df = pd.read_csv(input_csv_path)

# Extract the necessary columns for processing
inchi_column = df['InChI']
other_columns = df.drop(columns=['InChI'])

# Create an empty list to store the calculated descriptors
descriptor_list = []

# Initialize Mordred descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)

# Iterate over each InChI string
for inchi in inchi_column:
    # Convert the InChI string to an RDKit Mol object
    mol = Chem.MolFromInchi(inchi)

    # Check if the conversion was successful
    if mol is not None:
        # Calculate descriptors for the molecule using Mordred
        descriptors = calc(mol)

        # Convert the descriptors to a dictionary
        descriptors_dict = descriptors.asdict()

        # Handle non-serializable descriptors by converting them to a string representation
        descriptors_str = {k: str(v) for k, v in descriptors_dict.items()}

        # Append the descriptors dictionary to the list
        descriptor_list.append(descriptors_str)
    else:
        # If conversion failed, append an empty dictionary to the descriptor list
        descriptor_list.append({})

# Create a DataFrame for the descriptors
df_descriptors = pd.DataFrame(descriptor_list)

# Concatenate the original DataFrame (excluding the InChI column) and the descriptors DataFrame
df_final = pd.concat([other_columns, df_descriptors], axis=1)

# Save the final DataFrame with descriptors to a new CSV file
df_final.to_csv(output_csv_path, index=False)