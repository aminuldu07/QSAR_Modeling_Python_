# calculate descriptors with their nane as the column name using mordred from "smiles string"

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import time

# Read the CSV file
path = 'C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/U_SMILES1.csv'
df = pd.read_csv(path)

# Record the start time
start_time = time.time()

# Initialize Mordred descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)

# Create an empty DataFrame to store the descriptors
result_df = pd.DataFrame()

# Iterate over rows of the DataFrame
for index, row in df.iterrows():
    # Get the SMILES string for the current row
    smiles = row["SMILES1"]

    # Convert SMILES string to RDKit Mol object
    mol = Chem.MolFromSmiles(smiles)

    # Calculate descriptors for the molecule
    result = calc(mol)
    dic4 = result.asdict()

    # Create a temporary DataFrame from the descriptor dictionary
    temp_df = pd.DataFrame(dic4, index=[smiles])

    # Append the temporary DataFrame to the result DataFrame
    result_df = pd.concat([result_df, temp_df])

# Write the result DataFrame to a CSV file
result_df.to_csv("C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/mordred_descriptors_fromSMILES1_with_descriptors_name.csv")

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time, "seconds")