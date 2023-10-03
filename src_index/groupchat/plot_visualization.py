# filename: plot_visualization.py

import pandas as pd
import matplotlib.pyplot as plt

# download the CSV data
data = pd.read_csv('https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv')

# create a scatter plot of Weight vs Horsepower
plt.scatter(data['Weight'], data['Horsepower(HP)'])

# add labels and title
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('Relation between Weight and Horsepower')

# save the plot to a file
plt.savefig("weight_vs_horsepower.png")

print("Plot saved successfully as weight_vs_horsepower.png")