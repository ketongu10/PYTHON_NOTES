import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV




def as_main():

    """LOAD AND PREPARE DATA"""
    cancer_data = load_breast_cancer()
    df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
    df['target'] = cancer_data['target']

    X = df[cancer_data.feature_names].values
    y = df['target'].values

    """SET UP PARAMETERS TO FIND OPTIMAL VARIANT"""
    param_grid = {
        'n_estimators': [10, 25, 50, 75, 100],
    }

    """RANDOM FOREST SEARCHING THE BEST PARAMETERS"""
    rf = RandomForestClassifier(random_state=123)       # DEFINITION RF
    gs = GridSearchCV(rf, param_grid, scoring='f1', cv=5)       # DEFINITION PARAMETERS OPTIMISER
    gs.fit(X, y)

    """BEST PARAMETERS RESULTS"""
    print("best params:", gs.best_params_)
    print(list(range(1, 10)))


def features_weights():

    """LOAD AND PREPARE DATA"""
    cancer_data = load_breast_cancer()
    df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
    df['target'] = cancer_data['target']

    X = df[cancer_data.feature_names].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=101)

    """RANDOM FOREST WITH 10 TREES"""
    rf = RandomForestClassifier(n_estimators=10, random_state=111)
    rf.fit(X_train, y_train)

    """FEATURES IMPORTANCE"""
    ft_imp = pd.Series(rf.feature_importances_, index=cancer_data.feature_names).sort_values(ascending=False)
    print(ft_imp)

features_weights()