import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


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
        y[i] = (float(line.strip().split(',')[1]))
        i += 1
    return X, y


def plotData(X, y):
    X = X[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(X, y, marker='+')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


def plotDecisionBoundary(X, y, theta):
    plt.scatter(X[:, 1], y)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    x = range(4, 25)
    y = theta[0] + theta[1] * x
    plt.plot(x, y)
    plt.show()


def plotSurface(feat, y):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
            J_vals[i, j] = computeCost(feat, y, t)
    fig1 = plt.figure()
    # plot surface
    ax = fig1.add_subplot(211, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    # plot contour
    ax = fig1.add_subplot(111)
    ax.contour(theta0_vals, theta1_vals, J_vals)
    plt.show()


def plotCost(cost):
    plt.plot(range(1, len(cost)+1), cost)
    plt.xlabel("number of iterations")
    plt.ylabel("cost")
    plt.show()


def batchGradientDescent(X, y, learning_rate=0.01, iterations=1500):
    m = X.shape[0]
    theta = np.zeros((2, 1))
    cost = []
    for i in range(iterations):
        for j in range(2):
            theta[j, 0] = theta[j, 0] - learning_rate * (1.0 / m) * sum(
                (np.dot(X, theta) - y) * X[:, j].reshape((m, 1)))
        cost.append(computeCost(X, y, theta))
    return theta, cost


def computeCost(X, y, theta):
    m = X.shape[0]
    cost = 1.0 / (2 * m) * np.sum((np.dot(X, theta) - y) ** 2)
    return cost


def normalEquation(X, y):
    """
    calculate theta by normal equation
    """
    temp1 = np.linalg.inv(np.dot(X.transpose(), X))
    theta = np.dot(np.dot(temp1, X.transpose()), y)
    return theta


X, y = loadDataSet('ex1data1.txt')
# theta = normalEquation(X, y)  # two methods for calculating theta
theta, cost = batchGradientDescent(X, y, 0.01, 150)
plotCost(cost)
plotDecisionBoundary(X, y, theta)
plotSurface(X, y)
