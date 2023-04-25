import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

from sklearn.model_selection import GridSearchCV
from IPython.display import Image


def as_main():
    """DATA PREPARING"""
    df = pd.read_csv('./PANDAS/titanic.csv')
    df['male'] = df['Sex'] == 'male'
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=26)  # random_state used as seed


    """MODEL"""
    param_grid = {
        'max_depth': [5, 15, 25],
        'min_samples_leaf': [1, 3],
        'max_leaf_nodes': [10, 20, 34, 35, 50]}
    dt = DecisionTreeClassifier()
    gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
    gs.fit(X, y)
    print("best params:", gs.best_params_)


    """GRAPH PNG EXPORT"""
    """feature_names = ['Pclass', 'male']
    X = df[feature_names].values
    y = df['Survived'].values

    dt = DecisionTreeClassifier()
    dt.fit(X, y)

    dot_file = export_graphviz(dt, feature_names=feature_names)
    graph = graphviz.Source(dot_file)
    graph.render(filename='tree', format='png', cleanup=True)"""


