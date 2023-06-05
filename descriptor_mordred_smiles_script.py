import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

# Read the CSV file
df = pd.read_csv("U_SMILES.csv")

# Initialize Mordred descriptor calculator
calc = Calculator(descriptors, ignore_3D=True)

# Create an empty DataFrame to store the descriptors
result_df = pd.DataFrame()

# Iterate over rows of the DataFrame
for index, row in df.iterrows():
    # Get the SMILES string for the current row
    smiles = row["SMILES"]

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
result_df.to_csv("outputall.csv")

