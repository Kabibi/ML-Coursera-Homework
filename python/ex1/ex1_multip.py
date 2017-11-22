import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    m = len(lines)  # num of training examples
    n = len(lines[0].strip().split(',')) - 1    # num of features
    X = np.ones((m, n + 1)) # matrix of training examples with extra ones at the first column
    y = np.ones((m, 1)) # matrix of labels
    i = 0
    for line in lines:
        X[i, 1:] = map(float, line.strip().split(',')[:-1])
        y[i] = float(line.strip().split(',')[-1])
        i += 1
    return X, y


def featureNormalize(X):
    mean = []
    std = []
    for i in range(1, X.shape[1]):
        mean.append(np.mean(X[:, i]))
        std.append(np.std(X[:, i]))
        X[:, i] = (X[:, i] - mean[-1]) / std[-1]
    return X, mean, std # mean and std are needed for prediction


def batchGradientDescent(X, y, learning_rate=0.01, iterations=1500):
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros((n, 1))
    cost = []
    for i in range(iterations):
        for j in range(n):
            theta[j, 0] = theta[j, 0] - learning_rate * (1.0 / m) * sum(
                (np.dot(X, theta) - y) * X[:, j].reshape((m, 1)))
        cost.append(computeCost(X, y, theta))
    return theta, cost


def plotConvergence(iterations, cost):
    plt.plot(range(1, iterations + 1), cost)
    plt.show()


def computeCost(X, y, theta):
    m = X.shape[0]
    cost = 1.0 / (2 * m) * np.sum((np.dot(X, theta) - y) ** 2)
    return cost


X, y = loadDataSet('ex1data2.txt')
X, m, s = featureNormalize(X)
theta, cost = batchGradientDescent(X, y, 0.01, 1500)
x1 = float(raw_input("Enter the size of the house: "))
x2 = float(raw_input("Enter the number of bedrooms: "))
data = np.array([1, (x1 - m[0]) / s[0], (x2 - m[1]) / s[1]])
print "The predicted price is :" + str(np.ceil(np.dot(data, theta)[0]))
