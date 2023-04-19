import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

def get_y(x, model):
    return -(model.coef_[0][0]*x + model.intercept_[0]) / model.coef_[0][1]

def as_main():

    """DATA PREPARING"""
    df = pd.read_csv('./PANDAS/titanic.csv')
    df['male'] = df['Sex'] == 'male'
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=26) #random_state used as seed

    """MODEL FITTING"""
    model = LogisticRegression()
    model.fit(X_train, y_train)

    """PREDICTED RESULTS & EXAMPLES"""
    y_pred = model.predict(X_test)
    print(model.predict([[3, True, 22.0, 1, 0, 7.25]]))
    print(model.predict(X[:5]))
    print(model.predict_proba(X[:5])[:, 0])
    print(y[:5])
    print(model.coef_, model.intercept_)

    """ANALYSIS"""
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("precision:", precision_score(y_test, y_pred))
    print("recall:", recall_score(y_test, y_pred))
    print("f1 score:", f1_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred)) #CONFUSING MATRIX IS REVERCED IN SKLEARN

    """RENDERING"""
    plt.scatter(df['Fare'], df['Age'], c=df['Survived'])  # shows that 'c' eats array of ints
    xs = [40, 100]
    ys = [get_y(xs[0], model), get_y(xs[1], model)]
    plt.plot(xs, ys)
    plt.xlabel('Fare')
    plt.ylabel('Age')

    plt.show()



