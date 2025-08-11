import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
credit_card_data = pd.read_csv('creditcard.csv')

# First 5 rows of the dataset
print(credit_card_data.head())

# Dataset statistical description
print(credit_card_data.describe().transpose())

# Dataset info
print(credit_card_data.info())

# Check the number of missing values in each column
print(credit_card_data.isnull().sum())

# Histogram of features
credit_card_data.hist(figsize=(20, 20))
plt.show()

# --- WARNING: pairplot on the whole dataset is too slow! Use a sample. ---
# sns.pairplot(credit_card_data, hue='Class')
sns.pairplot(credit_card_data.sample(5000), hue='Class')  # safer for large data
plt.show()

# Standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(credit_card_data.drop(["Class"], axis=1)), columns=credit_card_data.drop(["Class"], axis=1).columns)
y = credit_card_data['Class']

print(X.head())

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Predictions and Evaluations
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Error rate for different values of K
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Final model evaluation with K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print('WITH k=1\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

from sklearn.metrics import ConfusionMatrixDisplay
conf_matrix = confusion_matrix(y_test, pred)
vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[True, False])
vis.plot()
plt.grid(False)
plt.show()
