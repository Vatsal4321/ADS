import pandas as pd
from sklearn.impute import KNNImputer

# Load the dataset
data = pd.read_csv('Energy_consumption.csv').head(10)

# Define a function for each imputation method
def impute_default(df, column, default_value):
    df[column] = df[column].fillna(default_value)
    print("Imputed using default value:", default_value)
    print(df.head())

def impute_previous_value(df, column):
    df[column] = df[column].ffill()
    print("Imputed using previous value")
    print(df.head())

def impute_next_value(df, column):
    df[column] = df[column].bfill()
    print("Imputed using next value")
    print(df.head())

def impute_mean(df, column):
    mean_value = df[column].mean()
    df[column] = df[column].fillna(mean_value)
    print("Imputed using mean value:", mean_value)
    print(df.head())

def impute_median(df, column):
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)
    print("Imputed using median value:", median_value)
    print(df.head())

def impute_mode(df, column):
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)
    print("Imputed using mode value:", mode_value)
    print(df.head())

def impute_max(df, column):
    max_value = df[column].max()
    df[column] = df[column].fillna(max_value)
    print("Imputed using max value:", max_value)
    print(df.head())

def impute_min(df, column):
    min_value = df[column].min()
    df[column] = df[column].fillna(min_value)
    print("Imputed using min value:", min_value)
    print(df.head())

def impute_most_frequent(df, column):
    most_frequent_value = df[column].value_counts().idxmax()
    df[column] = df[column].fillna(most_frequent_value)
    print("Imputed using most frequent value:", most_frequent_value)
    print(df.head())

def impute_knn(df, column):
    knn_imputer = KNNImputer(n_neighbors=5)
    df[column] = knn_imputer.fit_transform(df[[column]])
    print("Imputed using KNN")
    print(df.head())

# Choose the column you want to impute
column_to_impute = 'Temperature'

# Perform each type of imputation
impute_default(data.copy(), column_to_impute, default_value=25)
impute_previous_value(data.copy(), column_to_impute)
impute_next_value(data.copy(), column_to_impute)
impute_mean(data.copy(), column_to_impute)
impute_median(data.copy(), column_to_impute)
impute_mode(data.copy(), column_to_impute)
impute_max(data.copy(), column_to_impute)
impute_min(data.copy(), column_to_impute)
impute_most_frequent(data.copy(), column_to_impute)
impute_knn(data.copy(), column_to_impute)
