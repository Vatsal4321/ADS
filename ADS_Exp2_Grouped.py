import pandas as pd
import numpy as np

df = pd.read_csv('Energy_consumption.csv')

grouped_data = df.groupby('DayOfWeek')['EnergyConsumption']

# Central Tendency
mean_energy_consumption_grouped = grouped_data.mean()
median_energy_consumption_grouped = grouped_data.median()
mode_energy_consumption_grouped = grouped_data.apply(lambda x: x.mode()[0]) 

# Dispersion
range_energy_consumption_grouped = grouped_data.apply(lambda x: np.ptp(x))
variance_energy_consumption_grouped = grouped_data.var()
std_deviation_energy_consumption_grouped = grouped_data.std()

#skewness
skewness = grouped_data.skew()

# Print the results
print("Central Tendency:")
print("Mean Energy Consumption:")
print(mean_energy_consumption_grouped)
print("\nMedian Energy Consumption:")
print(median_energy_consumption_grouped)
print("\nMode Energy Consumption:")
print(mode_energy_consumption_grouped)

print("\nDispersion:")
print("Range of Energy Consumption:")
print(range_energy_consumption_grouped)
print("\nVariance of Energy Consumption:")
print(variance_energy_consumption_grouped)
print("\nStandard Deviation of Energy Consumption:")
print(std_deviation_energy_consumption_grouped)

print('\nSkewness of Dataset')
print("Skewness: ")
print(skewness)