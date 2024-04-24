import numpy as np

# Define the grouped data
grouped_values = ['0-5', '5-10', '10-15', '15-20', '20-25']
frequencies = [2, 6, 9, 3, 4]

# Calculate the midpoint for each group
midpoints = [(int(group.split('-')[0]) + int(group.split('-')[1])) / 2 for group in grouped_values]

# Calculate the total frequency
total_frequency = sum(frequencies)

# Calculate the mean
mean = sum(midpoint * frequency for midpoint, frequency in zip(midpoints, frequencies)) / total_frequency

# Calculate the deviation from the mean for each midpoint
deviations = [midpoint - mean for midpoint in midpoints]

# Calculate the sum of the products of frequencies and deviations
sum_of_products = sum(deviation * frequency for deviation, frequency in zip(deviations, frequencies))

# Calculate the sum of the squares of frequencies and deviations
sum_of_squares_frequencies = sum(frequency ** 2 for frequency in frequencies)
sum_of_squares_deviations = sum(deviation ** 2 for deviation in deviations)

# Calculate the correlation coefficient (Karl Pearson's coefficient of correlation)
correlation_coefficient = sum_of_products / np.sqrt(sum_of_squares_frequencies * sum_of_squares_deviations)

# Ensure the correlation coefficient is within the range [-1, 1]
correlation_coefficient = max(min(correlation_coefficient, 1), -1)

print("Karl Pearson's coefficient of correlation:", correlation_coefficient)

# Print the correlation matrix
correlation_matrix = np.array([[1, correlation_coefficient],
                                [correlation_coefficient, 1]])
print("\nCorrelation Matrix:")
print(correlation_matrix)
