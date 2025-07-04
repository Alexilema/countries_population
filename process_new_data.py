# -*- coding: utf-8 -*-

# Libraries
import os
import pandas as pd

from pathlib import Path
from sys import path

# Import our processing functions from script.py
func_path = r'C:\Users\alejandro.lema\Downloads\Challenge Data Engineer\script.py'
path.append(func_path)
import script as mf

#%%
# Specify the existing processed data file path

# For now, the code asumes that the data.csv file is on the same folder
# than this script.py file
script_path = str(Path(__file__).parent.resolve()) # this line fails when executed alone, the whole cell or script must be run
processed_path = os.path.join(script_path, 'data_processed.csv')

# Read existing data
df_pr = mf.read_data(
    processed_path,
    first_as_index=False)  # Careful, now first col is not the index

# Specify the NEW data file path
input_path = script_path

new_csv_filename = input("Especify the name of the .csv file to be processed."
                         "Please note it must be located at:\n"
                         f"{input_path}:"
                         )
# Make sure the extension is included
if not new_csv_filename.endswith('.csv'):
    new_csv_filename += '.csv'

input_path = os.path.join(input_path, new_csv_filename)

# Read the data
df = mf.read_data(input_path) # Here, first col is indeed the index

# Normalize the data
df = mf.normalize_mean_std(df)

# Remove boring features
df = mf.remove_boring(df)

# Get the population values from the REST countries API
df = mf.add_population(df)

# Include the new data in the existing df
df = pd.concat([df_pr, df])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# (Optional) Drop duplicates (duplicated in all columns)

# Disclaimer: after the process of normalization, the "duplicated" values may
# not be exactly identical bit by bit, causing drop_duplicates to remove less
# rows than expected / desired. To prevent this, we can previously round the
# values to a reasonable amount of decimals.
# df = df.round(6)

# # Or more professionally, we can compare the rows using a certain tolerante
# # (Pending to implement if needed)

# df = df.drop_duplicates()

# WARNING: Keep in mind that this may result in a loss of precision in the data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Finally, safe results
output_path = processed_path  # we want to update the existing .csv
df.to_csv(output_path, index=False)
