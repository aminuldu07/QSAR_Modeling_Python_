# To handle a CSV file with multiple columns, including the "InChI" column, the code as follows:

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd

# Read the input CSV file
df = pd.read_csv('INCHIKEY_to_inchi_rdeleted_543r.csv')

# Extract the necessary columns for processing
inchi_column = df['InChI']
other_columns = df.drop(columns=['InChI'])

# Create an empty list to store the calculated descriptors
descriptor_list = []

# Iterate over each InChI string
for inchi in inchi_column:
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

# Concatenate the original DataFrame (excluding the InChI column) and the descriptors DataFrame
df_final = pd.concat([other_columns, df_descriptors], axis=1)

# Save the final DataFrame with descriptors to a new CSV file
df_final.to_csv('descriptors.csv', index=False)




    
