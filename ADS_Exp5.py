import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Energy_consumption.csv')

# Filter out non-numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
numeric_data = data[numeric_columns]


# Bar Graph (Example: DayOfWeek vs. EnergyConsumption)
plt.figure(figsize=(10, 6))
sns.barplot(x='DayOfWeek', y='EnergyConsumption', data=data)
plt.title('Bar Graph of Average Energy Consumption by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Energy Consumption')
plt.show()

# Scatter Plot (Example: Temperature vs. EnergyConsumption)
plt.figure(figsize=(10, 6))
plt.scatter(data['Temperature'], data['EnergyConsumption'], alpha=0.5)
plt.title('Scatter Plot of Temperature vs. Energy Consumption')
plt.xlabel('Temperature')
plt.ylabel('Energy Consumption')
plt.show()

# Histogram (Example: EnergyConsumption)
plt.figure(figsize=(10, 6))
plt.hist(data['EnergyConsumption'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Energy Consumption Distribution')
plt.xlabel('Energy Consumption')
plt.ylabel('Frequency')
plt.show()

if len(numeric_columns) >= 3:
    plt.figure(figsize=(10, 6))
    plt.stackplot(data.index, [data[col] for col in numeric_columns[:3]], labels=numeric_columns[:3])
    plt.title('Stacked Plot of Numeric Columns')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
else:
    print("Insufficient numeric columns available for stack plot.")

# Box Plot (Example: Temperature)
plt.figure(figsize=(10, 6))
sns.boxplot(data['Temperature'])
plt.title('Box Plot of Temperature Distribution')
plt.xlabel('Temperature')
plt.show()

# Violin Plot (Example: Humidity)
plt.figure(figsize=(10, 6))
sns.violinplot(data['Humidity'], color='orange')
plt.title('Violin Plot of Humidity Distribution')
plt.xlabel('Humidity')
plt.show()

# Pie Chart (Example: Occupancy)
plt.figure(figsize=(10, 6))
occupancy_counts = data['Occupancy'].value_counts()
plt.pie(occupancy_counts, labels=occupancy_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Occupancy Distribution')
plt.show()

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Drop non-numeric columns if needed
data_numeric = data.select_dtypes(include=['number'])

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
