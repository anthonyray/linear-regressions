# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX",
"PTRATIO","B","LSTAT","MEDV"]

dat = pd.read_csv("housing/data.csv",delim_whitespace=True,skiprows=1,names=names)

price = np.reshape(dat['MEDV'].values,[dat['MEDV'].values.shape[0],1])
X = np.reshape(dat[['LSTAT','B','CRIM']].values,[dat['LSTAT'].values.shape[0],3])

price_train = price[:-20]
price_test = price[-20:]

X_train = X[:-20]
X_test = X[-20:]


# Different linear regression

clf = linear_model.LinearRegression(fit_intercept=False)

clf.fit(X_train,price_train)
print("OLS :")
print("Coeff : \n",clf.coef_)
print("Residual sum of squares: %.2f" % np.mean((clf.predict(X_train) - price_train) ** 2))
print('Variance score: %.2f' % clf.score(X_test, price_test))

n_alphas = 200
alphas = np.logspace(-10,2,n_alphas)

ridg = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    ridg.set_params(alpha=a)
    ridg.fit(X_train,price_train)
    coefs.append(ridg.coef_)

ax = plt.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
