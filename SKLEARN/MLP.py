from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_moons
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits



def as_main():
    """LOADS 8x8 HANDWRITTEN DIGITS"""
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    """MULTI-LAYER-PERCEPTRON FITTING"""
    mlp = MLPClassifier(random_state=2)
    mlp.fit(X_train, y_train)

    """RESULTS"""
    print(mlp.score(X_test, y_test))
    y_pred = mlp.predict(X_test)
    incorrect = X_test[y_pred != y_test]
    incorrect_true = y_test[y_pred != y_test]
    incorrect_pred = y_pred[y_pred != y_test]

    j = 0
    print(incorrect[j].reshape(8, 8).astype(int))
    print("true value:", incorrect_true[j])
    print("predicted value:", incorrect_pred[j])

    """RENDERING"""
    x = X_test[0]
    plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def draw():

    """LOADING MNIST 28x28 DATASET"""
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print(X.shape, y.shape)
    #print(np.min(X), np.max(X))
    #print(y[0:5])

    """SELECTING ONLY DIGITS <= 3"""
    X5 = X[y.astype('int') <= 3]    # y is a list of strings, we should reform it as list of ints
    y5 = y[y.astype('int') <= 3]

    """MULTI-LAYER-PERCEPTRON FITTING"""
    mlp = MLPClassifier(
        hidden_layer_sizes=(6,),
        max_iter=200, alpha=1e-4,
        solver='sgd', random_state=2)

    mlp.fit(X5, y5)

    """COEFFICIENTS"""
    print(len(mlp.coefs_))      # coefs_ and intercepts_ define a node
    print(mlp.coefs_)
    print(mlp.coefs_[0].shape, mlp.coefs_[1].shape)
    fig, axes = plt.subplots(2, 3, figsize=(5, 4))
    for i, ax in enumerate(axes.ravel()):
        coef = mlp.coefs_[0][:, i]  # 784 lines each contains 6 columns, we take one column per iteration
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(i + 1)
    plt.show()

draw()
