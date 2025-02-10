
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def var1():

    """DOESN'T WORK"""
    boston = load_boston()
    df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    df['target'] = boston.target
    print(boston['feature_names'])
    y = df['target'].values
    X = df[boston.feature_names].values

    # poly with x^2 and x^3
    polynomial = PolynomialFeatures(degree=3, include_bias=False)
    feat_poly = polynomial.fit_transform(X)

    regression = LinearRegression()
    model = regression.fit(feat_poly, y)
    print(model.score(feat_poly, y))


def var2():
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    # add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(8))

    svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svr_lin = SVR(kernel="linear", C=100, gamma="auto")
    svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
    lw = 2

    svrs = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ["RBF", "Linear", "Polynomial"]
    model_color = ["m", "c", "g"]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(
            X,
            svr.fit(X, y).predict(X),
            color=model_color[ix],
            lw=lw,
            label="{} model".format(kernel_label[ix]),
        )
        axes[ix].scatter(
            X[svr.support_],
            y[svr.support_],
            facecolor="none",
            edgecolor=model_color[ix],
            s=50,
            label="{} support vectors".format(kernel_label[ix]),
        )
        axes[ix].scatter(
            X[np.setdiff1d(np.arange(len(X)), svr.support_)],
            y[np.setdiff1d(np.arange(len(X)), svr.support_)],
            facecolor="none",
            edgecolor="k",
            s=50,
            label="other training data",
        )
        axes[ix].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=1,
            fancybox=True,
            shadow=True,
        )

    fig.text(0.5, 0.04, "data", ha="center", va="center")
    fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()

var2()