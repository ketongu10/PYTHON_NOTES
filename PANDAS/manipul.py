import pandas as pd


def as_main():
    df = pd.read_csv('./PANDAS/example.csv')
    print(df.head())
    print(df.describe())
    col = df[['Age', 'Fare']]
    to_numpy = df.values
    print(to_numpy)    #To array
    print(to_numpy[:, 1].sum())   #To select a column

def mask():
    df = pd.read_csv('./PANDAS/example.csv')

    arr = df[['Pclass', 'Fare', 'Age']].values[:10]
    mask = arr[:, 2] < 18
    print(arr[mask])    # ~ print i-th row if mask[i] is True
    print(arr[arr[:, 2] < 18])