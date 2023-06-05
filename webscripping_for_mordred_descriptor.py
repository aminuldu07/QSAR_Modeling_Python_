#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Get the current directory of the notebook file

import os

print(os.getcwd())


# Web Scrapping for getting the descriptors table from the mordred documentation

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Fetch the webpage
response = requests.get('https://mordred-descriptor.github.io/documentation/master/descriptors.html')
soup = BeautifulSoup(response.text, 'html.parser')

# Find the tables on the page
tables = soup.find_all('table')

# This list will hold all of the data
data = []

# Iterate over each table
for table in tables:
    # Find all rows in this table
    rows = table.find_all('tr')
    
    # Iterate over each row
    for row in rows:
        # Find all columns in this row
        cols = row.find_all('td')
        
        # Get the text from each column and add it to the list
        cols = [col.text.strip() for col in cols]
        data.append(cols)

# Create a pandas DataFrame from the data
df = pd.DataFrame(data, columns=["#", "Module", "Name", "Constructor", "Dimension", "Description"])

# Write the DataFrame to an Excel file
df.to_excel("mordred_descriptors.xlsx", index=False)

