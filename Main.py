import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from LogisticRegression import LogisticRegressionWithOptimization
from Plot import initial_plot, final_plot


if __name__ == "__main__":
    # load the data from the file
    initial_plot()
    # data = pd.read_csv("data2.txt", header=None)

    # X = data.iloc[:, :-1]

    # # labels are the last column of the data frame
    # y = data.iloc[:, -1]

    # # filter out the applicants that got admitted and the applicants that din't get admission
    # admitted = data.loc[y == 1]
    # not_admitted = data.loc[y == 0]

    # # get the numpy arrays to pass to the model
    # X = np.c_[np.ones((X.shape[0], 1)), X]
    # y = np.array(y)[:, np.newaxis]
    # theta = np.zeros((X.shape[1], 1))

    # # Logistic Regression from scratch using Gradient Descent
    # model = LogisticRegressionWithOptimization()
    # np.seterr(all='ignore', divide='ignore', over='ignore', under='ignore', invalid='ignore')
    # model.fit(X, y, theta)

    # # Compute the accuracy of the model
    # accuracy = model.accuracy(X, y.flatten())
    # parameters = model.weights
    # print("===========================================================")
    # print("The accuracy of the model in percentage is {}".format(accuracy))
    # print("===========================================================")

    # x_test = [1, 45, 85]
    # print("====================================================================================================================================")
    # print("Excepted probability of getting admission for a student with an Exam 1 score of 45 and an Exam 2 score of 85: ",model.predict(x_test))
    # print("====================================================================================================================================")

    # final_plot(X, parameters)
