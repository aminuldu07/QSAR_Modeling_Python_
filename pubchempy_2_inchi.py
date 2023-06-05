{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b311016b-a6e3-40fb-bd8b-f909c5f4ebbb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users\\mdaminulisla.prodhan\\All_My_Miscellenous\\pubchempy\n"
     ]
    }
   ],
   "source": [
    "# Get the current directory of the notebook file\n",
    "\n",
    "import os\n",
    "\n",
    "print(os.getcwd())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2626bd16-936c-4865-9a54-1e72a10df164",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Web Scrapping for getting the descriptors table from the mordred documentation\n",
    "\n",
    "import requests\n",
    "from bs4 import BeautifulSoup\n",
    "import pandas as pd\n",
    "\n",
    "# Fetch the webpage\n",
    "response = requests.get('https://mordred-descriptor.github.io/documentation/master/descriptors.html')\n",
    "soup = BeautifulSoup(response.text, 'html.parser')\n",
    "\n",
    "# Find the tables on the page\n",
    "tables = soup.find_all('table')\n",
    "\n",
    "# This list will hold all of the data\n",
    "data = []\n",
    "\n",
    "# Iterate over each table\n",
    "for table in tables:\n",
    "    # Find all rows in this table\n",
    "    rows = table.find_all('tr')\n",
    "    \n",
    "    # Iterate over each row\n",
    "    for row in rows:\n",
    "        # Find all columns in this row\n",
    "        cols = row.find_all('td')\n",
    "        \n",
    "        # Get the text from each column and add it to the list\n",
    "        cols = [col.text.strip() for col in cols]\n",
    "        data.append(cols)\n",
    "\n",
    "# Create a pandas DataFrame from the data\n",
    "df = pd.DataFrame(data, columns=[\"#\", \"Module\", \"Name\", \"Constructor\", \"Dimension\", \"Description\"])\n",
    "\n",
    "# Write the DataFrame to an Excel file\n",
    "df.to_excel(\"mordred_descriptors.xlsx\", index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5d7953d0-a4e8-46fc-abfe-14e4ae27e46b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generating files containing finally selected SMILES from U_SMILES.csv and SMILES and INCHIKEY from the \n",
    "# meged talbe on Approval_ID file\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "# Read the first CSV file \"U_SMILES.csv\"\n",
    "df_smiles = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/U_SMILES.csv')\n",
    "\n",
    "# Read the second CSV file \"merged_on_Approval_ID.csv\"\n",
    "df_merged = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/merged_on_Approval_ID.csv')\n",
    "\n",
    "# Merge the two dataframes based on the condition SMILES = SMILES1\n",
    "merged_df = pd.merge(df_smiles, df_merged, how='inner', left_on='SMILES1', right_on='SMILES')\n",
    "\n",
    "# Select the desired columns\n",
    "result_df = merged_df[['SMILES1', 'SMILES', 'INCHIKEY']]\n",
    "\n",
    "# Save the result to a new CSV file\n",
    "result_df.to_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/pubchempy.csv', index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "efa2bd7b-9a57-4eb2-8d33-f110808b961f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Deleting duplicate INCHIKEY from the \"merged_on_Approval_ID.csv\"\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "# Read the CSV file \"merged_on_Approval_ID.csv\"\n",
    "df_merged = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/merged_on_Approval_ID.csv')\n",
    "\n",
    "# Drop duplicate rows based on the \"INCHIKEY\" column\n",
    "unique_df = df_merged.drop_duplicates(subset='INCHIKEY')\n",
    "\n",
    "# Save the unique dataframe to a new CSV file\n",
    "unique_df.to_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/unique_inchikeys.csv', index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1a2fb897-cc04-4d2b-963c-cfd5dd149653",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking the effectiveness of deleting duplicate INCHIKEY by agian deleting the duplicate SMILES values \n",
    "import pandas as pd\n",
    "\n",
    "# Read the CSV file \"unique_inchikeys.csv\"\n",
    "df_unique_inchikeys = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/unique_inchikeys.csv')\n",
    "\n",
    "# Drop duplicate rows based on the \"SMILES\" column\n",
    "unique_smiles_df = df_unique_inchikeys.drop_duplicates(subset='SMILES')\n",
    "\n",
    "# Save the unique SMILES dataframe to a new CSV file\n",
    "unique_smiles_df.to_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/unique_smiles.csv', index=False)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b452f68b-1be8-46cf-8f3f-19ea83713031",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Final csv containing the identical SMILES and the corresponding INCHIKEY\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "# Read the CSV file \"unique_inchikeys.csv\"\n",
    "df_unique_inchikeys = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/unique_inchikeys.csv')\n",
    "\n",
    "# Read the CSV file \"U_SMILES.csv\"\n",
    "df_usmiles = pd.read_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/U_SMILES.csv')\n",
    "\n",
    "# Merge the dataframes based on the condition SMILES = SMILES1\n",
    "merged_df = pd.merge(df_unique_inchikeys, df_usmiles, how='inner', left_on='SMILES', right_on='SMILES1')\n",
    "\n",
    "# Select the desired columns\n",
    "result_df = merged_df[['SMILES1', 'SMILES', 'INCHIKEY']]\n",
    "\n",
    "# Save the result to a new CSV file\n",
    "result_df.to_csv('C:/Users/mdaminulisla.prodhan/All_My_Miscellenous/pubchempy/result_final.csv', index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "aaf2e32e-8b36-425d-a7a3-1bc33b71bfca",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert the INCHIKEY to inchi string using pubchempy package \n",
    "import pandas as pd\n",
    "import requests\n",
    "\n",
    "# Read the input CSV file\n",
    "df = pd.read_csv(r'C:\\Users\\mdaminulisla.prodhan\\All_My_Miscellenous\\pubchempy\\result_final.csv')\n",
    "\n",
    "# Create a new column for InChI\n",
    "df['InChI'] = ''\n",
    "\n",
    "# Iterate over the rows and retrieve InChI from PubChem using InChIKey\n",
    "for index, row in df.iterrows():\n",
    "    inchi_key = row['INCHIKEY']\n",
    "    try:\n",
    "        response = requests.get(f\"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/property/InChI/txt\")\n",
    "        if response.status_code == 200:\n",
    "            inchi = response.text.strip()\n",
    "            df.at[index, 'InChI'] = inchi\n",
    "        else:\n",
    "            df.at[index, 'InChI'] = 'Error'\n",
    "    except:\n",
    "        df.at[index, 'InChI'] = 'Error'\n",
    "\n",
    "# Save the updated DataFrame to a new CSV file\n",
    "df.to_csv(r'C:\\Users\\mdaminulisla.prodhan\\All_My_Miscellenous\\pubchempy\\result_final_with_inchi.csv', index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e7902d43-0287-408a-b6f2-e1b6bcb077ae",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
