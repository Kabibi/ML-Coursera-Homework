# coding=utf-8
"""
In this part of exercise we implement an anomaly detection
algorithm to detect anomalous behavior in server computers.
"""
import matplotlib.pyplot as plt
import scipy.io
import numpy as np


def loadDataSet(filename):
	data = scipy.io.loadmat(filename)
	X = data['X']
	Xval = data['Xval']
	yval = data['yval'].flatten()
	return X, Xval, yval


def estimateGaussian(X):
	"""
	compute mu and sigma
	"""
	mu = np.mean(X, axis=0)
	sigma2 = np.std(X, axis=0) ** 2
	return mu, sigma2


def multivariateGaussian(X, mu, sigma2):
	"""Computes the probability
	density function of the examples X under the multivariate gaussian
	distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
	treated as the covariance matrix. If Sigma2 is a vector, it is treated
	as the \sigma^2 values of the variances in each dimension (a diagonal
	covariance matrix)
	"""
	k = len(mu)

	if sigma2.ndim == 1:
		# convert sigma2 to a diagonal matrix
		sigma2 = np.diag(sigma2)

	# vectorized version of Multivariate Gaussian Distribution
	X = X - mu
	# p is a vector contains all probabilities of each examples
	p = (2 * np.pi) ** (- k / 2.0) * np.linalg.det(sigma2) ** (-0.5) * \
	    np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(sigma2)) * X, axis=1))

	return p


def visualizeFit(X, mu, sigma2):
	n = np.linspace(0, 35, 71)
	X1 = np.meshgrid(n, n)
	Z = multivariateGaussian(np.column_stack((X1[0].T.flatten(), X1[1].T.flatten())), mu, sigma2)
	Z = Z.reshape(X1[0].shape)

	plt.plot(X[:, 0], X[:, 1], 'bx')
	# Do not plot if there are infinities
	if not np.isinf(np.sum(Z)):
		plt.contour(X1[0], X1[1], Z, 10.0 ** np.arange(-20, 0, 3).T)


def selectThreshold(yval, pval):
	"""
	finds the best threshold to use for selecting outliers
	based on the results from a validation set (pval) and
	the ground truth (yval).
	"""
	bestEpsilon = 0
	bestF1 = 0
	stepsize = (np.max(pval) - np.min(pval)) / 1000

	for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
		predictions = (pval < epsilon) + 0
		tp = np.sum((yval == 1) & (predictions == 1))
		fp = np.sum((yval == 0) & (predictions == 1))
		fn = np.sum((yval == 1) & (predictions == 0))
		if tp + fp == 0:
			continue
		prec = float(tp) / (tp + fp)  # tips: cast int to float, or you will get 0
		rec = float(tp) / (tp + fn)
		F1 = 2.0 * prec * rec / (prec + rec)
		if F1 > bestF1:
			bestF1 = F1
			bestEpsilon = epsilon
	return bestEpsilon, bestF1


def part1():
	X, Xval, yval = loadDataSet('ex8data1.mat')
	mu, sigma2 = estimateGaussian(X)
	# get a list of probabilities
	p = multivariateGaussian(X, mu, sigma2)
	# visualize the fit
	visualizeFit(X, mu, sigma2)

	# use cross validation set(Xval) to select threshold.
	pval = multivariateGaussian(Xval, mu, sigma2)
	epsilon, F1 = selectThreshold(yval, pval)

	print('Best epsilon found using cross-validation: %e\n' % (epsilon))
	print('Best F1 on Cross Validation Set:  %f\n' % (F1))

	# Find the outliers in the training set and plot the
	outliers = np.nonzero(p < epsilon)[0]
	# Draw a red circle around those outliers
	plt.scatter(X[outliers, 0], X[outliers, 1], edgecolors='r', marker='o', s=200, facecolor='none')
	plt.xlabel("Latency(ms)")
	plt.ylabel("Throughput(mb/s")
	plt.show()


def part2():
	"""
	%% ================== Multidimensional Outliers ===================
    We will now use the code from the previous part and apply it to a
    harder problem in which more features describe each data point and only
    some features indicate whether a point is an outlier.
	"""
	X, Xval, yval = loadDataSet('ex8data2.mat')
	mu, sigma2 = estimateGaussian(X)
	p = multivariateGaussian(X, mu, sigma2)
	pval = multivariateGaussian(Xval, mu, sigma2)
	epsilon, F1 = selectThreshold(yval, pval)

	print('Best epsilon found using cross-validation: %e\n' % (epsilon))
	print('Best F1 on Cross Validation Set:  %f\n' % (F1))
	print('# Outliers found: %d\n' % (np.sum(p < epsilon)))
	print('(you should see a value epsilon of about 1.38e-18)\n\n')


def main():
	part1()


if __name__ == '__main__':
	main()
