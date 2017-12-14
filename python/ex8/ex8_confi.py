# coding=utf-8
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda):
	"""
	compute the value of the cost function given parameters X and Theta. (X and
	Theta are rolled into params and params must be the first parameter of this
	function. Because that is the convention if we want to optimize with fmin_cg
	provided by scipy.)
	"""
	# retrieve X and Theta from params
	X = params[:num_movies * num_features].reshape((num_movies, num_features))
	Theta = params[num_movies * num_features:].reshape((num_users, num_features))
	# Cost function with regularization term
	J = 0.5 * np.sum(R * ((X.dot(Theta.T) - Y) ** 2)) \
	    + _lambda / 2.0 * np.sum(Theta ** 2) \
	    + _lambda / 2.0 * np.sum(X ** 2)

	return J


def cofiCostFunGrad(params, Y, R, num_users, num_movies, num_features, _lambda):
	"""
	compute gradient of the cost function.
	"""
	X = params[:num_movies * num_features].reshape((num_movies, num_features))
	Theta = params[num_movies * num_features:].reshape((num_users, num_features))

	X_grad = np.zeros(X.shape)
	Theta_grad = np.zeros(Theta.shape)
	# My_X_grad = np.zeros(X.shape)
	# My_Theta_grad = np.zeros(Theta.shape)

	# compute X_grad with vectorization
	for i in range(X_grad.shape[0]):
		idx = np.nonzero(R[i, :])[0]
		theta_tmp = Theta[idx, :]
		y_tmp = Y[i, idx]
		x_tmp = X[i, :].reshape((1, X[i, :].shape[0]))
		X_grad[i, :] = (x_tmp.dot(theta_tmp.T) - y_tmp).dot(theta_tmp) + _lambda * X[i, :]
	# My_X_grad[i, :] = (theta_tmp.T.dot(theta_tmp.dot(x_tmp.T) -
	#                                    y_tmp.reshape((y_tmp.shape[0], 1)))).reshape(-1) + _lambda * X[i, :]

	# compute Theta_grad with vectorization
	for j in range(Theta_grad.shape[0]):
		idx = np.nonzero(R[:, j])[0]
		x_tmp = X[idx, :]
		y_tmp = Y[idx, j].reshape((Y[idx, j].shape[0], 1))
		theta_tmp = Theta[j, :].reshape((Theta[j, :].shape[0], 1))
		Theta_grad[j, :] = ((x_tmp.dot(theta_tmp)) - y_tmp).T.dot(x_tmp) + _lambda * Theta[j, :]
	# My_Theta_grad[j, :] = (x_tmp.T.dot((x_tmp.dot(theta_tmp)) - y_tmp)).T + _lambda * Theta[j, :]

	grad = np.append(X_grad.reshape(-1), Theta_grad.reshape(-1))
	return grad


def loadMovieList():
	"""
	load data from 'movie_idx.txt' and return a vector of names of all movies.
	"""
	n = 1682
	movieList = np.array(range(n), dtype='S100')
	fid = open('movie_ids.txt')
	lines = fid.readlines()
	m = 0
	for i in range(len(lines)):
		movieList[i] = lines[i].strip().split(' ', 1)[-1]
	return movieList


def normalizeRatings(Y, R):
	"""
	compute the mean of Y and the normalized Y excluded unrated movies
	"""
	m, n = Y.shape
	Ymean = np.zeros(m)
	Ynorm = np.zeros(Y.shape)
	for i in range(m):
		idx = np.nonzero(R[i, :] == 1)[0]
		Ymean[i] = np.mean(Y[i, idx])
		Ynorm[i, idx] = Y[i, idx] - Ymean[i]

	return Ymean, Ynorm


def main():
	"""
	%% =============== Part 1: Loading movie ratings dataset ================
	%  You will start by loading the movie ratings dataset to understand the
	%  structure of the data.
	"""
	data = scipy.io.loadmat('ex8_movies.mat')
	# %  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
	# %  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
	R = data['R']
	Y = data['Y']
	n_movies, n_users = R.shape
	n = 100

	print 'Average rating for movie 1 (Toy Story): %f\n\n' % (np.mean(Y[0, np.nonzero(R[0, :])]))
	plt.imshow(Y[:50, :50], extent=[0, 1, 0, 1])
	plt.xlabel('Movies')
	plt.ylabel('Users')
	plt.show()

	"""
	%% ============ Part 2: Collaborative Filtering Cost Function ===========
	%  You will now implement the cost function for collaborative filtering.
	%  To help you debug your cost function, we have included set of weights
	%  that we trained on that. Specifically, you should complete the code in
	%  cofiCostFunc.m to return J.
	"""
	num_users = 4
	num_movies = 5
	num_features = 3

	# Load data
	data = scipy.io.loadmat('ex8_movies.mat')
	R = data['R']
	Y = data['Y']

	data = scipy.io.loadmat('ex8_movieParams.mat')
	X = data['X']
	Theta = data['Theta']

	X = X[:num_movies, :num_features]  # n_m * n_f
	Theta = Theta[:num_users, :num_features]  # n_u * n_f
	Y = Y[:num_movies, :num_users]  # n_m * n*u
	R = R[:num_movies, :num_users]  # n_m * n_u
	params = np.append(X.reshape(-1), Theta.reshape(-1))

	# compute the value and the gradient of cost function
	J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
	grad = cofiCostFunGrad(params, Y, R, num_users, num_movies, num_features, 1.5)

	print('Cost at loaded parameters: %f\n\n' % (J))

	"""
	%% ============== Part 3: Entering ratings for a new user ===============
	%  Before we will train the collaborative filtering model, we will first
	%  add ratings that correspond to a new user that we just observed. This
	%  part of the code will also allow you to put in your own ratings for the
	%  movies in our dataset!
	"""
	movieList = loadMovieList()
	my_ratings = np.zeros(1682)
	my_ratings[1] = 4
	my_ratings[98] = 2
	my_ratings[7] = 3
	my_ratings[12] = 5
	my_ratings[54] = 4
	my_ratings[64] = 5
	my_ratings[66] = 3
	my_ratings[69] = 5
	my_ratings[183] = 4
	my_ratings[226] = 5
	my_ratings[355] = 5
	print('New user ratings:\n')
	for i in range(len(my_ratings)):
		if my_ratings[i] > 0:
			print ('Rated %d for %s\n' % (my_ratings[i], movieList[i]))

	"""
	%% ================== Part 4: Learning Movie Ratings ====================
	%  Now, you will train the collaborative filtering model on a movie rating 
	%  dataset of 1682 movies and 943 users
	"""
	print('\nTraining collaborative filtering...\n')
	data = scipy.io.loadmat('ex8_movies.mat')
	Y = data['Y']
	R = data['R']

	# convert my_ratings from (1682,) to (1682,1)
	my_ratings = my_ratings.reshape((my_ratings.shape[0], 1))

	# Add our own ratings to the data matrix
	Y = np.append(Y, my_ratings, axis=1)
	R = np.append(R, (my_ratings != 0), axis=1)

	# Normalize Ratings
	Ymean, Ynorm = normalizeRatings(Y, R)

	# Useful Values
	num_users = Y.shape[1]
	num_movies = Y.shape[0]
	num_features = 10

	# Set initial Parameters (Theta, X)
	X = np.random.randn(num_movies, num_features)
	Theta = np.random.randn(num_users, num_features)

	initial_parameters = np.append(X, Theta, axis=0)

	_lambda = 10
	theta = fmin_cg(f=cofiCostFunc, x0=initial_parameters, fprime=cofiCostFunGrad,
	                args=(Y, R, num_users, num_movies, num_features, _lambda), maxiter=100)

	X = theta[:num_movies * num_features].reshape((num_movies, num_features))
	Theta = theta[num_movies * num_features:].reshape((num_users, num_features))

	print('\n\nRecommender system learning completed.\n\n')

	"""
	%% ================== Part 5: Recommendation for you ====================
	%  After training the model, you can now make recommendations by computing
	%  the predictions matrix.
	"""
	p = X.dot(Theta.T)
	my_predictions = p[:, 1] + Ymean
	movieList = loadMovieList()

	top_movie_idx = my_predictions.argsort()[::-1]
	print ('\nTop recommendations for you: \n')

	for i in range(10):
		print ("prediction rating %.3f for movie %s\n" %
		       (my_predictions[top_movie_idx[i]], movieList[top_movie_idx[i]]))

	print ('\n\nOriginal ratings provided:\n')
	for i in range(len(my_ratings)):
		if my_ratings[i] > 0:
			print ("Rated %.2f for %s\n" % (my_ratings[i], movieList[i]))


if __name__ == '__main__':
	main()
