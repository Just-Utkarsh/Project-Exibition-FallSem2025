import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('creditcard.csv')
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    clf.fit(X_train_res, y_train_res)

    y_pred_probs = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auprc = average_precision_score(y_test, y_pred_probs)
    print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    new_tx_data = np.array([
        0, -1.35980713, -0.07278117, 2.53634673, 1.37815522,
        -0.33832077, 0.46238778, 0.23959855, 0.09869790, 0.36378697,
        0.09079417, -0.55159953, -0.61780086, -0.99138985, -0.31116935,
        1.46817697, -0.47040053, 0.20797124, 0.02579058, 0.40399296,
        0.25141210, -0.01830678, 0.27783758, -0.11047391, 0.06692807,
        0.12853936, -0.18911484, 0.13355838, -0.02105305, 149.62
    ])

    new_tx_df = pd.DataFrame([new_tx_data], columns=X.columns)

    new_tx_df[['Time', 'Amount']] = scaler.transform(new_tx_df[['Time', 'Amount']])

    proba = clf.predict_proba(new_tx_df)[:, 1][0]
    label = clf.predict(new_tx_df)[0]
    print(f"\nNew transaction fraud probability: {proba:.4f}")
    print("Prediction: ", "FRAUD" if label == 1 else "NOT FRAUD")

if __name__ == "__main__":
    main()
