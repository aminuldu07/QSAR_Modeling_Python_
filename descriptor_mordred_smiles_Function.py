# Calculate descriptor using Rdkit and mordred
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

def calculate_descriptors(filename):
    # Read the CSV file
    df = pd.read_csv(filename)

    # Initialize Mordred descriptor calculator
    calc = Calculator(descriptors, ignore_3D=True)

    # Create an empty list to store the descriptors
    descriptors_list = []

    # Iterate over rows of the DataFrame
    for index, row in df.iterrows():
        # Get the SMILES string for the current row
        smiles = row["SMILES"]

        # Convert SMILES string to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)

        # Calculate descriptors for the molecule
        result = calc(mol)
        dic4 = result.asdict()

        # Append the descriptor dictionary to the list
        descriptors_list.append(dic4)

    # Convert the list of descriptor dictionaries to a DataFrame
    descriptors_df = pd.DataFrame(descriptors_list)

    # Concatenate the original DataFrame with the descriptors DataFrame
    result_df = pd.concat([df, descriptors_df], axis=1)

    # Return the result DataFrame
    return result_df

# Use the function and save the result to a CSV file
result_df = calculate_descriptors("INCHIKEY_to_inchi_rdeleted_543r.csv")
result_df.to_csv("descriptors_INCHIKEY_to_inchi_rdeleted_543r.csv")

