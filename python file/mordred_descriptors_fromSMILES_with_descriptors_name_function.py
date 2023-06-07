# # calculate descriptors with their nane as the column name using mordred from "smiles string"

## main difference from the below function "Create an empty list to store the descriptors"
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import time

# Record the start time
start_time = time.time()

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

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time, "seconds")



###############@@@@@@@@@@@@@@@@@@@@@@@@@ SECOND FUNCTION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

## main difference from the above function "Create an empty DataFrame to store the descriptors"

def calculate_descriptors(filename):
    # Read the CSV file
    df = pd.read_csv(filename)

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

    # Return the result DataFrame
    return result_df
result_df = calculate_descriptors("U_SMILES.csv")
result_df.to_csv("output.csv")