# coding=utf-8
"""
0. 本代码是手写数字的识别, 只是训练出了一个分类器模型.
1. 分类器在training set上的精确度只有不到80%, 仍然需要改进.
2. 主要分类错误的地方在数字'8'的分类上, 目前还不知道为什么.
   在训练时发现一下奇怪的现象:(使用fmin_bfgs)
    Warning: Desired error not necessarily achieved due to precision loss.
         Current function value: 0.000000
         Iterations: 0
         Function evaluations: 14
         Gradient evaluations: 3
    这说明当把'8'和'其他数字'这两个类进行训练时, 没有找到最优化的theta.目前还没解决该问题.
    但是使用fmin_cg时是可以找到最优的theta的, 但是debug时发现对'8'识别的正确率很低.
3. 另外, initial_theta的选取和优化函数的选取也会对精确度造
   成很大的影响.
4. fmin_bfgs和fmin_cg都会出现类似"Warning: Desired error
   not necessarily achieved due to precision loss."的警告
"""

import numpy as np
import scipy.special
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs, fmin_cg


def loadDataSet(filename):
	"""
	There are 5000 training examples in ex3data1.mat, where each training
	example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is
	represented by a floating point number indicating the grayscale intensity at
	that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional
	vector. Each of these training examples becomes a single row in our data
	matrix X. This gives us a 5000 by 400 matrix X where every row is a training
	example for a handwritten digit image.
	"""
	mat = scipy.io.loadmat(filename)  # 5000 training examples, 20*20=400 pixels
	m, n = mat['X'].shape
	X = np.ones((m, n + 1))
	X[:, 1:] = mat['X']

	y = mat['y']
	for i in range(len(mat['y'])):
		if y[i] == 10:
			y[i] = 0

	return X, y


def displayData(X, example_width=None):
	if not example_width:
		example_width = np.round(np.sqrt(X.shape[1])).astype(np.int64)  # np.round() return float64

	m, n = X.shape
	example_height = (n / example_width).astype(np.int64)

	display_rows = np.floor(np.sqrt(m)).astype(np.int64)
	display_cols = np.ceil(m / display_rows).astype(np.int64)

	print display_rows, display_cols

	pad = 1

	display_array = - np.ones((pad + display_rows * (example_height + pad),
	                           pad + display_cols * (example_width + pad)))

	curr_ex = 1
	for j in range(display_rows):
		for i in range(display_cols):
			if curr_ex > m:
				break

			max_val = np.max(np.abs(X[curr_ex, :]))
			display_array[np.ix_(pad + (j - 1) * (example_height + pad) + range(example_height),
			                     pad + (i - 1) * (example_width + pad) + range(example_width))] = \
				X[curr_ex, :].reshape(example_height, example_width) / max_val

			curr_ex = curr_ex + 1

		if curr_ex > m:
			break

	plt.imshow(display_array.T, cmap=plt.cm.gray)
	plt.show()


def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))


def lrCostFunction(theta, X, y, _lambda):
	"""
	1. regularization项的求和从1开始, 不包括theta[0]
	2. 如果y.shape=(5000,), h.shape=(5000,1) => (y*h).shape=(5000,5000)
	   矩阵运算一定要搞清楚shape, 本人花了不少时间在shape的debug上
	3. -y.dot(np.log(h)) != (-y).dot(np.log(h)) 能打括号绝不手软!
	4. 向量的内积有两种写法: x.dot(y) 和np.sum(x*y), 注意都要是向量, 而且维度正确
	"""
	m, n = X.shape
	h = sigmoid(X.dot(theta))
	return 1.0 / m * np.sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h)) + \
	       _lambda / (2.0 * m) * (np.sum((theta ** 2)[1:]))


def gradient(theta, X, y, _lambda):
	"""
	grad[0] 单独拎出来计算
	"""
	m, n = X.shape
	grad = X.T.dot(sigmoid(X.dot(theta)) - y) / m
	grad[1:] = grad[1:] + ((theta[1:] * _lambda) / m)
	return grad


def oneVsAll(X, y, _lambda, K=10):
	m, n = X.shape
	# initial_theta = np.zeros((1, n))    # get poor performance
	initial_theta = np.random.rand(1, n)  # get better performance
	theta = np.zeros((K, n))

	for i in range(0, K):  # K classifiers
		y_temp = (y == i).reshape(-1)
		theta[i, :] = fmin_cg(lrCostFunction, initial_theta, fprime=gradient, args=(X, y_temp, _lambda), disp=True)
		# theta[i, :] = fmin_bfgs(lrCostFunction, initial_theta, fprime=gradient, args=(X, y_temp, _lambda))  # get poor performance

	return theta


def predictOneVsAll(theta, X, y):
	"""
	calculate training accuracy
	"""
	hypothesis = sigmoid(X.dot(theta.T))  # 5000*10 matrix
	result = np.argmax(hypothesis, axis=1)

	res = (result == y.reshape(-1))
	accuracy = np.sum(res) / float(len(res))
	return accuracy


def __main__():
	X, y = loadDataSet('ex3data1.mat')
	_lamda = 0
	theta = oneVsAll(X, y, _lamda)
	print "Predict accuracy: %.2f" % (predictOneVsAll(theta, X, y))


__main__()
