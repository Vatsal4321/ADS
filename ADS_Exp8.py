import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv("Energy_consumption.csv")
y = data['Occupancy']

# Encoding the target variable based on a threshold (set your own threshold)
threshold = y.median()  # This is an arbitrary threshold for demonstration purposes
y_binned = y.apply(lambda x: 'High' if x >= threshold else 'Low')
y_encoded = LabelEncoder().fit_transform(y_binned)

# Count of each class before SMOTE
class_labels = ['Low', 'High']  # This is your encoded class labels
class_counts_before = pd.Series(y_encoded).value_counts().to_dict()
print("\nCount of each class before SMOTE:")
print("Low Energy Consumption:", class_counts_before[0])
print("High Energy Consumption:", class_counts_before[1])

# Plot the distribution of the target variable before SMOTE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(y_encoded, bins=2, color='lightblue', edgecolor='darkblue')
plt.title('Distribution of Energy Consumption Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=class_labels)

# Selecting only numeric attributes for SMOTE
numeric_columns = data.select_dtypes(include=['number']).columns
X_numeric = data[numeric_columns].drop('EnergyConsumption', axis=1)

# Apply SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_numeric, y_encoded)

# Count of each class after SMOTE
class_counts_after = pd.Series(y_resampled).value_counts().to_dict()
print("\nCount of each class after SMOTE:")
print("Low Energy Consumption:", class_counts_after[0])
print("High Energy Consumption:", class_counts_after[1])

# Plot the distribution of the target variable after SMOTE
plt.subplot(1, 2, 2)
plt.hist(y_resampled, bins=2, color='lightgreen', edgecolor='darkgreen')
plt.title('Distribution of Energy Consumption After SMOTE')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=class_labels)

plt.tight_layout()
plt.show()