# coding=utf-8
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def kMeansInitCentroids(X, K):
	"""
	Randomly select K examples from training examples as the initial K centroids
	"""
	centroids = np.zeros((K, X.shape[1]))  # K centroids
	index = []

	# Randomly select K different centroids
	while len(index) != K:
		tmp = np.random.random_integers(0, X.shape[0] - 1)
		if tmp not in index:
			index.append(tmp)

	centroids[:, :] = X[index, :]

	return centroids


def findClosetCentroids(X, centroids):
	"""
	Compute the distance between training examples and all the centroids.
	The idx tells which cluster the training examples belong to.
	"""
	m, n = X.shape
	K = centroids.shape[0]
	idx = np.zeros(m)  # m

	for i in range(m):
		temp = np.tile(X[i, :], K).reshape(centroids.shape)
		idx[i] = np.argmin(np.sum((centroids - temp) ** 2, axis=1))
	return idx


def computeMeans(X, idx, K):
	"""
	Given assignments of every point to a centroid, the second phase of the
	algorithm recomputes, for each centroid, the mean of the points that were
	assigned to it.
	"""
	m, n = X.shape
	centroids = np.zeros((K, n))
	count = np.zeros(K)

	for j in range(m):
		centroids[int(idx[j])] += X[j]

	for i in range(m):
		count[int(idx[i])] += 1

	return centroids / np.tile(count.reshape((K, 1)), n)


def runKmeans(X, K, iterations=20):
	centroids = kMeansInitCentroids(X, K)
	for iter in range(iterations):
		idx = findClosetCentroids(X, centroids)  # 第二次迭代的时候idx的值竟然全为9
		centroids = computeMeans(X, idx, K)
	return centroids, idx


def plotResult(X, K, centroids, idx):
	# Implementation tips:
	# What's the difference between [[] for x in range(3)] and [[]]*3 ?
	# [[]]*3 will give you the same result like above but
	# the list are not distinct instances,they are just n
	# references to the same instance.

	temp_X1 = [[] for _ in range(K)]
	temp_X2 = [[] for _ in range(K)]

	for i in range(X.shape[0]):
		temp_X1[int(idx[i])].append(X[i, 0])  # temp_X1[i] save the X1 of cluster i.
		temp_X2[int(idx[i])].append(X[i, 1])  # save X2[i] save the X2 of cluster i.

	for i in range(K):
		plt.scatter(temp_X1[i], temp_X2[i], s=200)  # plot points
		plt.scatter(centroids[i, 0], centroids[i, 1], marker='+', s=200, c='b')  # plot centroids

	plt.show()


def compressImage(K):
	"""
	Use K-Means to compress image
	"""
	A = plt.imread('bird_small.png')
	m = A.shape[0] * A.shape[1]
	X = np.zeros((m, A.shape[2]))

	# build matrix X
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			X[i + j * A.shape[0], :] = A[i, j, :]

	centroids, idx = runKmeans(X, K)

	for i in range(X.shape[0]):
		X[i, :] = centroids[int(idx[i]), :]

	compressedImage = np.zeros(A.shape)
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			compressedImage[i, j, :] = X[i + j * A.shape[0], :]

	displayImage(compressedImage)


def displayImage(imageMat):
	plt.imshow(imageMat)


def main():
	plt.figure()
	for i in range(1, 17):
		plt.subplot(4, 4, i)
		compressImage(K=i)
		plt.title("K = " + str(i))
	plt.show()


if __name__ == '__main__':
	main()
