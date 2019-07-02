#polynomial regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('Position_Salaries.csv')
X=data.iloc[:,1:2].values
y=data.iloc[:,2:].values

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)

#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures           #used to just create a quadritic sequence
polyreg=PolynomialFeatures(degree=2)
X_poly=polyreg.fit_transform(X)

linreg2=LinearRegression()                         #applying regression modle to quadritic model
linreg2.fit(X_poly,y)

#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures           #used to just create a higher order sequence
polyreg=PolynomialFeatures(degree=5)
X_pol=polyreg.fit_transform(X)

linreg3=LinearRegression()                         #applying regression modle to quadritic model
linreg3.fit(X_pol,y)

#visulisation of linear regression
plt.scatter(X, y, color='red')
plt.plot(X,linreg.predict(X), color='blue')
plt.title("bluff or real")
plt.xlabel("Level")
plt.ylabel("salary")
plt.show() 

#visulisaion of quadritic regression
plt.scatter(X, y, color='red')
plt.plot(X,linreg2.predict(X_poly), color='blue')
plt.title("bluff or real")
plt.xlabel("Level")
plt.ylabel("salary")
plt.show() 

#visulisaion of higher order regression
plt.scatter(X, y, color='red')
plt.plot(X,linreg3.predict(X_pol), color='blue')
plt.title("bluff or real")
plt.xlabel("Level")
plt.ylabel("salary")
plt.show() 