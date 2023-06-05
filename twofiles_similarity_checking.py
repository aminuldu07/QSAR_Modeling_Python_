# CHECK WHETHER TWO CSV  FLIES ARE Identical 

import pandas as pd

def compare_csv_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Making sure both DataFrames are sorted in the same way
    df1 = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2 = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)

    # Checking if the two DataFrames are identical
    are_identical = df1.equals(df2)
    
    return are_identical
is_identical = compare_csv_files("outputall.csv", "outputfunc.csv")
if is_identical:
    print("The files are identical.")
else:
    print("The files are different.")
