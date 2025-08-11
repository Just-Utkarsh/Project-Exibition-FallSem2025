import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
credit_card_data = pd.read_csv('creditcard.csv')

# Show first and last 5 rows
print(credit_card_data.head())
print(credit_card_data.tail())

# Dataset information
print(credit_card_data.info())

# Check for missing values
print(credit_card_data.isnull().sum())

# Distribution of legit and fraudulent transactions
print(credit_card_data['Class'].value_counts())

# Separate datasets
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print("Legit shape:", legit.shape)
print("Fraud shape:", fraud.shape)

# Statistical measures
print("Legit transaction amount stats:")
print(legit.Amount.describe())

print("Fraudulent transaction amount stats:")
print(fraud.Amount.describe())

# Compare means for both classes
print("Feature means by class:")
print(credit_card_data.groupby('Class').mean())

# Undersample legit transactions to match fraud count
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

print("New dataset head:")
print(new_dataset.head())
print("New dataset tail:")
print(new_dataset.tail())
print("Class distribution in new dataset:")
print(new_dataset['Class'].value_counts())
print("Feature means in new dataset:")
print(new_dataset.groupby('Class').mean())

# Split features and target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print("Features (X):")
print(X)
print("Target (Y):")
print(Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("Full dataset shape:", X.shape)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on Training data:', training_data_accuracy)

# Evaluate on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on Test Data:', test_data_accuracy)
