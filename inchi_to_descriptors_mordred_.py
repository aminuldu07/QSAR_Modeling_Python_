# It seems that the df_final DataFrame contains columns with dictionaries as values, 
# which cannot be directly saved to a CSV file using the to_csv() method. To overcome this,
# you can convert the dictionaries to strings before saving the DataFrame to a CSV file. 

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import json

# Read the input CSV file
df = pd.read_csv('INCHIKEY_to_inchi_rdeleted_543r.csv')

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

        # Convert the descriptors dictionary to a string
        descriptors_str = json.dumps(descriptors.asdict())

        # Append the descriptors string to the list
        descriptor_list.append(descriptors_str)
    else:
        # If conversion failed, append an empty string to the descriptor list
        descriptor_list.append('')

# Create a DataFrame for the descriptors
df_descriptors = pd.DataFrame(descriptor_list, columns=['Descriptors'])

# Concatenate the original DataFrame (excluding the InChI column) and the descriptors DataFrame
df_final = pd.concat([other_columns, df_descriptors], axis=1)

# Save the final DataFrame with descriptors to a new CSV file
df_final.to_csv('descriptors.csv', index=False)
