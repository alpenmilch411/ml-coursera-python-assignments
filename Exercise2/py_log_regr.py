# imports
import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize

# Loading data

path_data = os.path.join('Data', 'ex2data1.txt')
feature_n = 2
def load_data(path_data, feature_n):
	data = np.loadtxt(path_data, delimiter=',')
	X, y = data[:, 0:feature_n], data[:, feature_n]
	return X,y

def plotData(X, y):
	''' Plots data oiunts X and y into a new figure. Use * for positive
		examples and use o for negative examples.'''

	# Create Mask to deterimine pos/neg examples for X based on y
	pos = y == 1
	neg = y == 0

	# Create new figure
	fig = pyplot.figure()

	''' Explanation of X[pos/neg,0/1]:
	Using the true/false mask from pos/neg numpy takes all values which are
	from X and takes the first/second column -> Features'''
    
	pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
	pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
	
	# Labeling
	pyplot.xlabel('X1')
	pyplot.ylabel('X2')
	pyplot.legend(['True', 'False'])
	
	pyplot.show()


def sigmoid(z):
	''' Define Sigmoid Function
	The sigmoid function normalizes data so that the output of the
	hypothesis function h_theta(x)= theta.T*x is between 0 and 1.
	The hypothesis function thus outputs the probability (range 0-1)
	that the classification based on theta and x is true.
	'''

	# convert input to a numpy array


	#compute sigmoid output
	g = 1/(1+np.exp(-z))
	return g


def costFunction(theta, X, y):
	''' Cost function computes the cost and gradient of X relative to y
	for a given set of theta parameters'''

	h = sigmoid(X.dot(theta))

	J = (np.log(h).T.dot(-y)-np.log(1-h).T.dot((1-y)))/m
	grad = (1/m) * X.T.dot(h-y)
	return J, grad


def plotDecisionBoundary(plotData, theta, X, y):
	theta = np.array(theta)
	plotData(X[:, 1:3], y)
	if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
		plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
		plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
		pyplot.plot(plot_x, plot_y)

        # Legend, specific for the exercise
		pyplot.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
		pyplot.xlim([30, 100])
		pyplot.ylim([30, 100])
		pyplot.show()
	else:
        # Here is the grid range
		u = np.linspace(-1, 1.5, 50)
		v = np.linspace(-1, 1.5, 50)

		z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
		for i, ui in enumerate(u):
			for j, vj in enumerate(v):
				z[i, j] = np.dot(mapFeature(ui, vj), theta)

		z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
		pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
		pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)
		pyplot.show()

def predict(theta,X):
	#Evaluate using sigmoid for optimized theta parameters
	p = sigmoid(X.dot(theta))
	
	#1 or 0 
	p = p >= 0.5
	return p

def accuracy(prediction):
	return np.mean(prediction == y) * 100
	pass




# Load data and split into X and y
X, y = load_data(path_data, feature_n)

#Add intercept to x
m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)

#initial theta values
init_theta = np.zeros(n+1)





#Optimize theta parameters using scipy.optimize

#set max number of iterations
options = {'maxiter': 400}

res = optimize.minimize(
	costFunction,
	init_theta,
	(X, y),
	jac=True,
	method='TNC',
	options=options
	)


#Return cost of cost function at optimized theta paramers
cost = res.fun

#Return optimized theta
theta = res.x

#Printing cost at optimized theta & respective theta parameters
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Theta found by optimize.minimize \t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))

prediction = predict(theta, X)

accura = accuracy(prediction)

print('Train Accuracy: {:.2f} %'.format(accura))







