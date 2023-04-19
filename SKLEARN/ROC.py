import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


def as_main():

    """DATA PREPARING"""
    df = pd.read_csv('./PANDAS/titanic.csv')
    df['male'] = df['Sex'] == 'male'
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # random_state used as seed

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

    """THE CLOSER ROC_CURVE TO LEFT UPPER CORNER THE BETTER MODEL IS"""
    print("AUC score:", roc_auc_score(y_test, y_pred_proba[:, 1]))

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    plt.show()
