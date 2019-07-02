#multiple linear regression

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data set
data=pd.read_csv("50_Startups.csv")
X=data.iloc[:,0:4].values
y=data.iloc[:,4:].values

#encoding the string type
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder_X=OneHotEncoder(categorical_features=[3])
X=onehotencoder_X.fit_transform(X).toarray()

#scaling dummy variable
X=X[:,1:]

#creating training ans test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

#fitting to training set
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)

regression.score(X_train,y_train)    #to get R^2 of model

#predictiing the result
y_predict=regression.predict(X_test)

#optimizing model using backward elimination
import statsmodels.formula.api as mp
X=np.append(arr =np.ones((50,1),dtype=int,),values=X,axis=1)

                             #iteration to perform backard propagation
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=mp.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_ols=mp.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,3,4,5]]
regressor_ols=mp.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,3,5]]
regressor_ols=mp.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
regressor_ols.score(y,X_opt)

X_opt=X[:,[0,5]]
regressor_ols=mp.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()