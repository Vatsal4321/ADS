import pandas as pd
import numpy as np
df = pd.read_csv('Energy_consumption.csv')
energy_consumption = df['EnergyConsumption']

# Central Tendency
mean_energy_consumption = np.mean(energy_consumption)
median_energy_consumption = np.median(energy_consumption)
mode_energy_consumption = energy_consumption.mode()[0]

# Dispersion
range_energy_consumption = np.ptp(energy_consumption)
variance_energy_consumption = np.var(energy_consumption)
std_deviation_energy_consumption = np.std(energy_consumption)

#skewness
skewness = energy_consumption.skew()

print("Central Tendency:")
print(f"Mean Energy Consumption: {mean_energy_consumption:.2f}")
print(f"Median Energy Consumption: {median_energy_consumption:.2f}")
print(f"Mode Energy Consumption: {mode_energy_consumption:.2f}")

print("\nDispersion:")
print(f"Range of Energy Consumption: {range_energy_consumption:.2f}")
print(f"Variance of Energy Consumption: {variance_energy_consumption:.2f}")
print(f"Standard Deviation of Energy Consumption: {std_deviation_energy_consumption:.2f}")

print('\nSkewness of Dataset')
print(f"Skewness: ", skewness)
