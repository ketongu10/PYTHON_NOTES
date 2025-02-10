import matplotlib.pyplot as plt
import pandas as pd


def as_main():
    df = pd.read_csv('./PANDAS/example.csv')
    df['male_int'] = [1 if i == ' male' else 0 for i in (df['Sex'])]
    print(df['male_int'])
    plt.scatter(df['Age'], df['Fare'], c=df['male_int']) #shows that 'c' eats array of ints
    plt.plot([0, 80], [85, 5])
    plt.xlabel('Age')
    plt.ylabel('Fare')

    plt.show()