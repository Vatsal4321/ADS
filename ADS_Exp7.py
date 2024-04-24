import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detect_anomaly(data_array, k=2, threshold=1):
    anomalies = []
    for i, point in enumerate(data_array):
        distances = np.abs(data_array - point)
        distances.sort()
        k_nearest = distances[1:k+1]
        mean_distance = np.mean(k_nearest)
        if mean_distance > threshold:
            anomalies.append((i, point))  # Store the index along with the point
    return anomalies

# Load your dataset
df = pd.read_csv('Energy_consumption.csv')

# Choose a variable to analyze for anomalies (e.g., 'EnergyConsumption')
energy_consumption_values = df['EnergyConsumption'].values

# Detect anomalies using the detect_anomaly function
anomalies = detect_anomaly(energy_consumption_values, k=2, threshold=2)

# Extract the indices and values of anomalies for plotting
anomaly_indices, anomaly_values = zip(*anomalies)

# Extract normal data points and their indices
normal_indices = [i for i in range(len(energy_consumption_values)) if energy_consumption_values[i] not in anomaly_values]
normal_points = [energy_consumption_values[i] for i in normal_indices]

# Plot normal points and anomalies with Energy Consumption on the x-axis and Index on the y-axis
plt.figure(figsize=(10, 6))
plt.plot(normal_points, normal_indices, 'o', label='Normal Points')
plt.plot(anomaly_values, anomaly_indices, 'o', color='red', label='Anomalies')
plt.xlabel('Energy Consumption')
plt.ylabel('Index')
plt.title('Energy Consumption Anomalies')
plt.legend()
plt.show()
