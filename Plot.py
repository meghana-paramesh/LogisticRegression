import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def initial_plot():
    data = pd.read_csv("data2.txt", header=None)
    X = data.iloc[:, :-1]

    y = data.iloc[:, -1]
    plt.figure(0)

    admitted = data.loc[y == 1]
    not_admitted = data.loc[y == 0]

    plt.xlabel('Exam 1 score', fontsize=8)
    plt.ylabel('Exam 2 score', fontsize=8)
    print(admitted.iloc[:, 0])

    # plots
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], c="black", marker="+", linewidth=0.5, s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], c="yellow", marker="o", edgecolors='black', linewidth=0.5, s=10,
                label='Not Admitted')
    plt.legend(loc="upper right", prop={'size': 6})

    plt.savefig('initial_plot.png')

def final_plot(X, parameters):
    x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
    y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]

    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.legend(loc="upper right", prop={'size': 6})
    plt.savefig('final_plot.png')
