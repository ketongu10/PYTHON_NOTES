from sklearn.model_selection import KFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

def as_main():

    """DATA PREPARING"""
    df = pd.read_csv('./PANDAS/titanic.csv')
    df['male'] = df['Sex'] == 'male'
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values

    scores = []

    """DATA SPLITTING"""
    kf = KFold(n_splits=5, shuffle=True)    # splits data on 5 series with 4:1 train-test ratio. kf returns indexes!!!
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
        print(model.coef_, model.intercept_)
    print(scores)
    print(np.mean(scores))

    final_model = LogisticRegression()
    final_model.fit(X, y)