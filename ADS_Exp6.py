import numpy as np
from scipy.stats import norm

temperature_data = [20, 21, 19, 22, 23]  
energy_consumption_data = [30, 33, 27, 28, 32] 

h0 = input("Enter the null hypothesis (H0): ")
h1 = input("Enter the alternative hypothesis (H1): ")

x1 = np.mean(temperature_data)  
x2 = np.mean(energy_consumption_data)  
std_dev_1 = np.std(temperature_data)  
std_dev_2 = np.std(energy_consumption_data) 
n = len(temperature_data)  # Sample size

Z_score = (x1 - x2) / np.sqrt((std_dev_1**2 / n) + (std_dev_2**2 / n))

alpha = 0.05
critical_value = norm.ppf(1 - alpha/2)

p_value = 2 * (1 - norm.cdf(np.abs(Z_score)))

print("Critical value:", critical_value)
print("P-value:", p_value)

if abs(Z_score) > critical_value:
    print("Reject the null hypothesis (H0)")
else:
    print("Fail to reject the null hypothesis (H0)")





    