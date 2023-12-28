import numpy as np
from scipy.optimize import minimize


class LogisticRegressionWithOptimization:

    def sigmoid(self, x):
        # Sigmoid Activation function
        return 1 / (1 + np.exp(-x))

    def net_input(self, theta, x):
        # Computes the weighted sum of inputs
        return np.dot(x, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        proability = self.probability(theta, x)
        total_cost = -(1 / m) * np.sum(
            y * np.log(proability) + (1 - y) * np.log(
                1 - proability))

        if np.array_equiv(theta, np.zeros((x.shape[1], 1))):
            print("===========================================================")
            print("Initial cost_function value: ",total_cost)
            print("===========================================================")
        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

    def fit(self, x, y, theta):
        opt_parameters = minimize(self.cost_function, theta, method='BFGS', jac=self.gradient,
               args=(x, y.flatten()))
        self.weights = opt_parameters.x
        print("===========================================================")
        print("Optimal Parameters: ")
        print("===========================================================")
        print("optimal cost function: ",opt_parameters.fun)
        print("Optimal theta values: ",self.weights)
        print("Total number of iterations to reach the optimal value: ",opt_parameters.nit)
        return self

    def predict(self, x):
        theta = self.weights[:, np.newaxis]
        return self.probability(theta, x)

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100