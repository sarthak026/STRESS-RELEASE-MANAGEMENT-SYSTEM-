# Plot the distribution of stress levels
sns.countplot(x='StressLevel', data=data)
plt.title('Distribution of Stress Levels')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Select features and target variable
features = ['Age', 'Gender', 'HeartRate', 'SleepHours']
target = 'StressLevel'

X = data[features]
y = data[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
