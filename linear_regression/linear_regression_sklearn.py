from __future__ import division, print_function, unicode_literals
import numpy as np
from sklearn import datasets, linear_model

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)

print("Solution found by scikit-learn: ", regr.coef_)

