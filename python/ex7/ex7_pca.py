# coding=utf-8
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ex7


def loadDataSet(filename):
	X = scipy.io.loadmat(filename)['X']
	return X


def plotData(X, mu, U, S):
	"""
	plot the scatter and the principal component
	"""
	plt.scatter(X[:, 0], X[:, 1])
	point1 = mu + 1.5 * S[0] * U[:, 0]
	point2 = mu + 1.5 * S[1] * U[:, 1]
	plt.plot([mu[0], point1[0]], [mu[1], point1[1]])
	plt.plot([mu[0], point2[0]], [mu[1], point2[1]])


def featureNormalize(X):
	m, n = X.shape
	mean = np.tile(np.mean(X, axis=0), m).reshape(X.shape)
	std = np.tile(np.std(X, axis=0), m).reshape(X.shape)
	return (X - mean) / std, mean[0, :], std[0, :]


def pca(X):
	sigma = 1.0 / X.shape[0] * X.T.dot(X)
	U, S, V = np.linalg.svd(sigma, full_matrices=True)
	return U, S


def projectData(X, U, K):
	U_reduce = U[:, 0:K]
	return X.dot(U_reduce)


def recoverData(Z, U, k):
	return Z.dot(U[:, :k].T)


def displayData(X, example_width=None):
	"""
	Given a matrix X of multiple images, display all the images.
	"""
	m, n = X.shape
	width = int(np.floor(np.sqrt(m * n) + np.sqrt(m)))
	dispImage = np.zeros((width, width))
	for i in range(m):
		start_x = int(np.floor(i / (m ** 0.5))) * int(n ** 0.5)
		start_y = (i % int(m ** 0.5)) * int(n ** 0.5)
		dispImage[start_x:start_x + int(n ** 0.5), start_y:start_y + int(n ** 0.5)] = X[i, :].reshape((32, 32))
	plt.imshow(dispImage.T)


def part1():
	X = loadDataSet('ex7data1.mat')
	k = 1
	plt.figure()

	# Before running PCA, it is important to first normalize X
	X_norm, mu, sigma = featureNormalize(X)
	# Run PCA
	U, S = pca(X_norm)
	# plot data and the principal components
	plotData(X, mu, U, S)
	plt.show()

	Z = projectData(X_norm, U, k)
	X_recover = recoverData(Z, U, k)
	# error
	print X_recover - X_norm


def part2():
	K = 100
	X = loadDataSet('ex7faces.mat')

	plt.subplot(121)
	displayData(X[:100, :])
	X_norm, mu, sigma = featureNormalize(X)
	U, S = pca(X_norm)
	Z = projectData(X_norm, U, K)
	X_rec = recoverData(Z, U, K)

	plt.subplot(122)
	displayData(X_rec[:100, :])
	plt.show()


def part3():
	A = plt.imread('bird_small.png')
	A = A / 255
	img_size = A.shape
	X = A.reshape(img_size[0] * img_size[1], 3)
	K = 16
	max_iters = 10
	centroids, idx = ex7.runKmeans(X, K, max_iters)

	# plot 3D figure
	fig = plt.figure()
	axis = fig.add_subplot(121, projection='3d')
	axis.scatter(X[:1000, 0], X[:1000, 1], X[:1000, 2], s=50, c=idx[:1000], marker='o')

	# dimension reduction
	X_norm, mu, sigma = featureNormalize(X)
	U, S = pca(X_norm)
	Z = projectData(X_norm, U, 2)
	axis = fig.add_subplot(122)
	plt.scatter(Z[:1000, 0], Z[:1000, 1], c=idx[:1000], marker='o')
	plt.show()


def main():
	part1()
	part2()
	part3()


if __name__ == '__main__':
	main()
