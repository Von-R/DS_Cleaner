import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pyampute.exploration.mcar_statistical_tests import MCARTest

file_path = 'path/to/yourfile.csv'
data = pd.read_csv(file_path)

# EDA: Explatory Data Analysis


# Data Cleaning

# Get numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Get categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Conduct Little's MCAR Test
mt = MCARTest(method="little")
p_value = mt.little_mcar_test(data)
print("Little's MCAR test p-value:", p_value)

# Choose either to drop NaN or impute them
data.dropna(inplace=True) # Remove rows with NaN values
# OR
# Impute numerical columns with mean
imputer_num = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer_num.fit_transform(data[numerical_columns])

# Impute categorical columns with the most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = imputer_cat.fit_transform(data[categorical_columns])


data = pd.get_dummies(data, columns=categorical_columns)

scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

z_scores = stats.zscore(data[numerical_columns]) # Apply z-score only to numerical columns
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]

def custom_clean(data):
    # Your custom cleaning code here
    return data

data = custom_clean(data)