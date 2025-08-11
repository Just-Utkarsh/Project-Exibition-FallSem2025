import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('creditcard.csv')
print(df.columns)
print(df.head(3))
print(df.info())
print(df.describe())
print(df.isnull().sum())
# -- Pairplot (optional: may be slow on large data) --
# Consider sampling for visualization
# sns.pairplot(df.sample(5000), hue='Class', palette='Set1')
# plt.show()

from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

from sklearn import tree

plt.figure(figsize=(20, 25))
tree.plot_tree(
    dtree,
    feature_names=X.columns,
    class_names=['Class-1', 'Class-0'],
    rounded=True,
    filled=True,
    proportion=True
)
plt.show()



