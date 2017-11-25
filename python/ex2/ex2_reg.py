'''
This is the regularized logistic regression. We want to predict
whether microchips from a fabrication plant passes quality
assurance.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs


def loadDataSet(filename):
    """
    We insert 1s in the first column of X for itercept term
    """
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


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))  # use np.exp instead of math.exp


def costFunction(theta, X, y, lamda):
    """
    Cost function with regularization term.
    """
    h = sigmoid(np.dot(X, theta.reshape((len(theta), 1))))
    m = X.shape[0]
    return (1.0 / m) * np.sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h)) + lamda / (2.0 * m) * np.sum(theta ** 2)


def gradient(theta, X, y, lamda):
    """
    return f' which is one argument passed to fmin_bfgs
    """
    h = sigmoid(np.dot(X, theta.reshape((len(theta), 1))))  # h theta of x
    m = X.shape[0]  # number of training examples
    theta[0] = (1.0 / m) * np.sum((h - y) * X[:, 0].reshape((len(X[:, 0]), 1)))  # check dimension!
    for j in range(1, X.shape[1]):
        theta[j] = (1.0 / m) * np.sum((h - y) * X[:, j].reshape((len(X[:, j]), 1))) + lamda / m * theta[
            j]  # check dimension!
    return theta


def mapFeature(X1, X2):
    """
    This function maps the two input features to quadratic
    features used in the regularization exercise.
    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    Inputs X1, X2 must be the same size
    """
    degree = 6
    out = np.ones((X1.shape[0],
                   (degree + 1) * (degree + 2) / 2))  # we map the two features to (degree+1)*(degree+2)/2 features
    index = 0
    for i in range(0, degree):
        for j in range(0, degree - i + 1):
            out[:, index] = (X1 ** i) * (X2 ** j)
            index += 1
    return out


def plotDecisionBoundary(theta, X, y, lamda):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta
    PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    positive examples and o for the negative examples. X is assumed to be a either
    1) Mx3 matrix, where the first column is an all-ones column for the intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    """
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
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.title('lambda =  ' + str(lamda))
    plt.legend([line1, line2], ['y=1', 'y=0'])
    u = np.linspace(-1, 1.5, 50)  # x axis
    v = np.linspace(-1, 1.5, 50)  # y axis
    z = np.zeros((len(u), len(v)))  # we need to generate z(x, y)

    for i in range(1, len(u)):
        for j in range(1, len(v)):
            z[i, j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])), theta)  # check type
    np.array(u).reshape((len(u), 1))

    plt.contour(u, v, z.transpose())


def analysis():
    """
    plot decision boundary given different lambda
    """
    plt.figure()
    start_pos = 221
    lamda = [0, 10, 100, 1000]
    initial_theta = np.zeros((X_new.shape[1]))

    for i in range(len(lamda)):
        plt.subplot(start_pos)
        start_pos += 1
        theta = fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X_new, y, lamda[i]))
        plotDecisionBoundary(theta, X, y, lamda[i])
    plt.show()


def predict(theta, X):
    out = mapFeature(X[:, 1], X[:, 2])
    return (np.dot(out, theta)) >= 0


X, y = loadDataSet('ex2data2.txt')
# map the original features to polynomial features
X_new = mapFeature(X[:, 1], X[:, 2])
initial_theta = np.zeros((X_new.shape[1]))

# plot the decision boundary of different given lambda
analysis()

# make predictions
theta = fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X_new, y, 1))
X_predict, _ = loadDataSet('predicting.txt')
print predict(theta, X)
