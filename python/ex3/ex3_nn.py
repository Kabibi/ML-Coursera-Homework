import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math


def predict(Theta1, Theta2, X):
	# Theta1 = 25*401
	# Theta2 = 10*26
	# X = 5000*400

	# PREDICT Predict the label of an input given a trained neural network
	#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	#   trained weights of a neural network (Theta1, Theta2)

	# turns 1D X array into 2D
	if X.ndim == 1:  # n dimensions
		X = np.reshape(X, (-1, X.shape[0]))

	# Useful values
	m = X.shape[0]
	num_labels = Theta2.shape[0]

	# You need to return the following variables correctly
	p = np.zeros((m, 1))

	# ====================== YOUR CODE HERE ======================
	# Instructions: Complete the following code to make predictions using
	#               your learned neural network. You should set p to a
	#               vector containing labels between 1 to num_labels.
	#

	# add column of ones as bias unit from input layer to second layer
	X = np.column_stack((np.ones((m, 1)), X))  # = a1

	# calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
	a2 = sigmoid(np.dot(X, Theta1.T))

	# add column of ones as bias unit from second layer to third layer
	a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))

	# calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
	a3 = sigmoid(np.dot(a2, Theta2.T))

	# get indices as in predictOneVsAll
	p = np.argmax(a3, axis=1)

	# =========================================================================

	return p + 1  # offsets python's zero notation


def displayData(X, example_width=None):
	# DISPLAYDATA Display 2D data in a nice grid
	#   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
	#   stored in X in a nice grid. It returns the figure handle h and the
	#   displayed array if requested.

	# closes previously opened figure. preventing a
	# warning after opening too many figures
	plt.close()

	# creates new figure
	plt.figure()

	# turns 1D X array into 2D
	if X.ndim == 1:
		X = np.reshape(X, (-1, X.shape[0]))

	# Set example_width automatically if not passed in
	if not example_width or not 'example_width' in locals():
		example_width = int(round(math.sqrt(X.shape[1])))

	# Gray Image
	plt.set_cmap("gray")

	# Compute rows, cols
	m, n = X.shape
	example_height = n / example_width

	# Compute number of items to display
	display_rows = int(math.floor(math.sqrt(m)))
	display_cols = int(math.ceil(m / display_rows))

	# Between images padding
	pad = 1

	# Setup blank display
	display_array = -np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

	# Copy each example into a patch on the display array
	curr_ex = 1
	for j in xrange(1, display_rows + 1):
		for i in xrange(1, display_cols + 1):
			if curr_ex > m:
				break

			# Copy the patch

			# Get the max value of the patch to normalize all examples
			max_val = max(abs(X[curr_ex - 1, :]))
			rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
			cols = pad + (i - 1) * (example_width + pad) + np.array(range(example_width))

			# Basic (vs. advanced) indexing/slicing is necessary so that we look can assign
			# 	values directly to display_array and not to a copy of its subarray.
			# 	from stackoverflow.com/a/7960811/583834 and
			# 	bytes.com/topic/python/answers/759181-help-slicing-replacing-matrix-sections
			# Also notice the order="F" parameter on the reshape call - this is because python's
			#	default reshape function uses "C-like index order, with the last axis index
			#	changing fastest, back to the first axis index changing slowest" i.e.
			#	it first fills out the first row/the first index, then the second row, etc.
			#	matlab uses "Fortran-like index order, with the first index changing fastest,
			#	and the last index changing slowest" i.e. it first fills out the first column,
			#	then the second column, etc. This latter behaviour is what we want.
			#	Alternatively, we can keep the deault order="C" and then transpose the result
			#	from the reshape call.
			display_array[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = np.reshape(X[curr_ex - 1, :],
			                                                                       (example_height, example_width),
			                                                                       order="F") / max_val
			curr_ex += 1

		if curr_ex > m:
			break

	# Display Image
	h = plt.imshow(display_array, vmin=-1, vmax=1)

	# Do not show axis
	plt.axis('off')

	plt.show(block=False)

	return h, display_array


def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))


def __main__():
	mat = scipy.io.loadmat('ex3weights.mat')
	mat_data = scipy.io.loadmat('ex3data1.mat')
	X = mat_data['X']
	y = mat_data['y']
	m, n = X.shape
	Theta1 = mat["Theta1"]
	Theta2 = mat["Theta2"]

	## ================= Part 3: Implement Predict =================
	#  After training the neural network, we would like to use it to predict
	#  the labels. You will now implement the "predict" function to use the
	#  neural network to predict the labels of the training set. This lets
	#  you compute the training set accuracy.

	pred = predict(Theta1, Theta2, X)

	print('Training Set Accuracy: {:f}'.format((np.mean(pred == y) * 100)))

	rp = np.random.permutation(m)

	for i in range(m):
		# Display
		displayData(X[rp[i], :])

		pred = predict(Theta1, Theta2, X[rp[i], :])
		print('Neural Network Prediction: {:d} (digit {:d})'.format(pred[0], (pred % 10)[0]))


__main__()
