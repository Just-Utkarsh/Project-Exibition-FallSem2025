import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
credit_card_data = pd.read_csv('creditcard.csv')

# Show columns, first and last rows, dataset info
print("Columns:", credit_card_data.columns.tolist())
print("Head:\n", credit_card_data.head())
print("Tail:\n", credit_card_data.tail())
print("Info:")
print(credit_card_data.info())
print("Missing values per column:\n", credit_card_data.isnull().sum())

# Count class distribution
print("Class distribution:\n", credit_card_data['Class'].value_counts())

# Drop 'Time' feature (generally not useful for modeling)
credit_card_data = credit_card_data.drop("Time", axis=1)

# Standard scale 'Amount' feature, drop original column
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
credit_card_data['std_Amount'] = scaler.fit_transform(credit_card_data['Amount'].values.reshape(-1, 1))
credit_card_data = credit_card_data.drop("Amount", axis=1)

# Count plot for class distribution
sns.countplot(x="Class", data=credit_card_data)
plt.show()

# Handle class imbalance by random undersampling (to 1:2 fraud:legit ratio)
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy=0.5)
cols = credit_card_data.columns.tolist()
cols = [c for c in cols if c != "Class"]
X = credit_card_data[cols]
y = credit_card_data["Class"]
X_under, y_under = undersample.fit_resample(X, y)

# Visualize before and after undersampling
fig, axs = plt.subplots(ncols=2, figsize=(13, 4.5))
sns.countplot(x="Class", data=credit_card_data, ax=axs[0])
sns.countplot(x="Class", data=pd.DataFrame({'Class': y_under}), ax=axs[1])
fig.suptitle("Class repartition before and after undersampling")
axs[0].set_title("Before")
axs[1].set_title("After")
plt.show()

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2, random_state=1)

# Train SVM model (enable probability estimates for ROC/PR curve)
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve

model = SVC(probability=True, random_state=2)
model.fit(X_train, y_train)
y_pred_svm = model.predict(X_test)

# Evaluation metrics
print("Accuracy SVM:", metrics.accuracy_score(y_test, y_pred_svm))
print("Precision SVM:", metrics.precision_score(y_test, y_pred_svm))
print("Recall SVM:", metrics.recall_score(y_test, y_pred_svm))
print("F1 Score SVM:", metrics.f1_score(y_test, y_pred_svm))

# Confusion Matrix
matrix_svm = confusion_matrix(y_test, y_pred_svm)
cm_svm = pd.DataFrame(matrix_svm, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])
sns.heatmap(cm_svm, annot=True, cmap="Blues", fmt='g')
plt.title("Confusion Matrix SVM")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.show()

# ROC and AUC
y_pred_svm_proba = model.predict_proba(X_test)[:,1]
fpr_svm, tpr_svm, _ = metrics.roc_curve(y_test, y_pred_svm_proba)
auc_svm = metrics.roc_auc_score(y_test, y_pred_svm_proba)
print("AUC SVM:", auc_svm)

plt.plot(fpr_svm, tpr_svm, label=f"SVM, AUC={auc_svm:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC curve')
plt.legend(loc=4)
plt.show()

# Precision-Recall curve
svm_precision, svm_recall, _ = precision_recall_curve(y_test, y_pred_svm_proba)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(svm_recall, svm_precision, color='orange', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()
