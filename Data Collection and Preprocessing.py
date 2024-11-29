import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('')

# Display first few rows of the dataset
print(data.head())

# Data Cleaning
# Handling missing values
data = data.dropna()

# Removing duplicates
data = data.drop_duplicates()

# Convert categorical features to numerical
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Normalize numerical features
scaler = StandardScaler()
data[['Age', 'HeartRate', 'SleepHours']] = scaler.fit_transform(data[['Age', 'HeartRate', 'SleepHours']])

# Display dataset after preprocessing
print(data.head())
