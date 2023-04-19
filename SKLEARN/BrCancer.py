import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression



def as_main():
    cancer_data = load_breast_cancer()

    df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
    df['target'] = cancer_data['target']
    y = df['target'].values
    X = df[cancer_data.feature_names].values
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    model.predict([X[0]])
    print(model.predict([X[0]]))
    print(model.score(X, y))

def smth():
    from sklearn.linear_model import LogisticRegression
    from matplotlib import pyplot as plt
    import numpy as np
    n = int(input())
    X = []
    for i in range(n):
        X.append([float(x) for x in input().split()])
    X = np.array(X)
    y = np.array([int(x) for x in input().split()])
    datapoint = [float(x) for x in input().split()]

    model = LogisticRegression()
    model.fit(X, y)
    print(model.predict([datapoint])[0])
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
