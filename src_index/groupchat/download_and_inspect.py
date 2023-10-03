# filename: download_and_inspect.py

import pandas as pd

# download the CSV data
data = pd.read_csv('https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv')

# print the fields in the dataset
print("Fields in the dataset: ", data.columns.tolist())