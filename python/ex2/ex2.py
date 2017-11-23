"""
We implemented logistic regression in this code.
We provided two different methods for getting the optimal theta which can minimize the cost function.
One is using gradient descent. The other is to use the fmin_bfgs provided by Scipy package.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs


def loadDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    m = len(lines)
    n = len(lines[0].strip().split(',')) - 1
    X = np.ones((m, n + 1))
    y = np.ones((m, 1))
    i = 0
    for line in lines:
        X[i, 1:] = map(float, line.strip().split(',')[:-1])
        y[i, 0] = float(line.strip().split(',')[-1])
        i += 1
    return X, y


def plotDecisionBoundary(X, y, theta):
    m = X.shape[0]
    pos = []
    neg = []
    for i in range(m):
        if y[i] == 1:
            pos.append(list(X[i, 1:]))
        else:
            neg.append(list(X[i, 1:]))
    line1 = plt.scatter(np.array(pos)[:, 0], np.array(pos)[:, 1], c='r', marker='.', s=200)
    line2 = plt.scatter(np.array(neg)[:, 0], np.array(neg)[:, 1], c='b', marker='+', s=200)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend([line1, line2], ['Admitted', 'Not admitted'])

    x_axis = np.linspace(-2, 2, 100)
    # y_axis = -(theta[0] + theta[1] * (x_axis - m[0]) * s[0]) / theta[2]*s[1] + m[1]
    y_axis = -theta[0] / theta[2] - theta[1] / theta[2] * x_axis
    plt.plot(x_axis, y_axis)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))  # use np.exp instead of math.exp


def featureNormalize(X):
    """
    Avoid overflow
    """
    mean = []
    std = []
    for column in range(1, X.shape[1]):
        mean.append(X[:, column].mean())
        std.append(X[:, column].std())
        X[:, column] = (X[:, column] - mean[-1]) / std[-1]
    return X, mean, std


def costFunction(theta, X, y):
    """
    Cost function which we want find the minimum
    """
    h = sigmoid(np.dot(X, theta.reshape((len(theta), 1))))
    m = X.shape[0]
    return (1.0 / m) * np.sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))


def gradient(theta, X, y):
    """
    return f' which is one argument passed to fmin_bfgs
    """
    h = sigmoid(np.dot(X, theta.reshape((len(theta), 1))))
    m = X.shape[0]
    for j in range(X.shape[1]):
        theta[j] = (1.0 / m) * np.sum((h - y) * X[:, j].reshape((len(X[:, j]), 1)))  # check dimension!
    return theta


def gradientDescent(X, y, learning_rate=0.01, iterations=1500):
    """
    calculate theta by gradient descent
    """
    theta = np.zeros(X.shape[1])
    m = X.shape[0]
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta.reshape((len(theta), 1))))
        for j in range(X.shape[1]):
            theta[j] = theta[j] - learning_rate * (1.0 / m) * np.sum((h - y) * X[:, j].reshape((len(X[:, j]), 1)))
    return theta


X, y = loadDataSet('ex2data1.txt')
X, m, s = featureNormalize(X)

initial_theta = np.zeros(X.shape[1])

theta1 = gradientDescent(X, y, 0.001, 15000)
theta2 = fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X, y))

plotDecisionBoundary(X, y, theta1)
plotDecisionBoundary(X, y, theta2)

plt.show()
