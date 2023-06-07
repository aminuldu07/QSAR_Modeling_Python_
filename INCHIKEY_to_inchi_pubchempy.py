# Creating a df having columns ( SMILES, SMILES1,INCHIKEY) from "merged_on_Approval_ID.csv" and "U_SMILES.csv"
# Converting INCHIKEY to "inchi string"

# Initial processing of the csv file ("merged_on_Approval_ID.csv") for getting unique INCHIKEY/SMILES
# Deleting duplicate INCHIKEY from the "merged_on_Approval_ID.csv"
# unique_inchikeys

import pandas as pd

# Read the CSV file "merged_on_Approval_ID.csv"
df_merged = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/merged_on_Approval_ID.csv')

# Drop duplicate rows based on the "INCHIKEY" column
unique_df = df_merged.drop_duplicates(subset='INCHIKEY')

# Save the unique dataframe to a new CSV file
unique_df.to_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/unique_inchikeys.csv', index=False)

# Checking the effectiveness of deleting duplicate INCHIKEY by agian deleting the duplicate SMILES values
# and checking the differnce between these two files (two files should be identical)
# unique_smiles

# Read the CSV file "unique_inchikeys.csv"
df_unique_inchikeys = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/unique_inchikeys.csv')

# Drop duplicate rows based on the "SMILES" column
unique_smiles_df = df_unique_inchikeys.drop_duplicates(subset='SMILES')

# Save the unique SMILES dataframe to a new CSV file
unique_smiles_df.to_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/unique_smiles.csv', index=False)

# Determine the differnce between the "unique_inchikeys.csv" and "unique_smiles.csv"
# Both file should be indentical

def compare_csv_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Making sure both DataFrames are sorted in the same way
    df1 = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2 = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)

    # Checking if the two DataFrames are identical
    are_identical = df1.equals(df2)
    
    return are_identical

is_identical = compare_csv_files("unique_inchikeys.csv", "unique_smiles.csv")
if is_identical:
    print("The files are identical.")
else:
    print("The files are different.")
    
# Generating files containing finally selected SMILES from "U_SMILES.csv" and SMILES and 
# INCHIKEY from the "unique_inchikeys.csv"
# SMILES = SMILES1


# Read the first CSV file "U_SMILES.csv"
df_smiles = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/U_SMILES.csv')

# Read the second CSV file "merged_on_Approval_ID.csv"
df_unique_inchikeys = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/unique_inchikeys.csv')


# Merge the two dataframes based on the condition SMILES = SMILES1
merged_df = pd.merge(df_smiles, df_unique_inchikeys, how='inner', left_on='SMILES1', right_on='SMILES')

# Select the desired columns
result_df = merged_df[['SMILES1', 'SMILES', 'INCHIKEY']]

# Save the result to a new CSV file
result_df.to_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/smile_smiles1_Inchikey.csv', index=False)


##### Convert the INCHIKEY to inchi string using pubchempy package 

import requests

# Read the input CSV file
df = pd.read_csv(r'C:\Users\mdaminulisla.prodhan\All_My_Miscellenous\pubchempy\smile_smiles1_Inchikey.csv')

# Create a new column for InChI
df['InChI'] = ''

# Iterate over the rows and retrieve InChI from PubChem using InChIKey
for index, row in df.iterrows():
    inchi_key = row['INCHIKEY']
    try:
        response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/property/InChI/txt")
        if response.status_code == 200:
            inchi = response.text.strip()
            df.at[index, 'InChI'] = inchi
        else:
            df.at[index, 'InChI'] = 'Error'
    except:
        df.at[index, 'InChI'] = 'Error'

# Save the updated DataFrame to a new CSV file
df.to_csv(r'C:\Users\mdaminulisla.prodhan\All_My_Miscellenous\pubchempy\INCHIKEY_to_inchi.csv', index=False)

