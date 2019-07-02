#simple linear regression

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt

#impoting main data file
data_set=pd.read_csv("Salary_Data.csv")
X=data_set.iloc[:,0:1].values                   
Y=data_set.iloc[:,1:2].values

#splitting into training and test set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(X,Y,test_size=0.20,random_state=0)

#fitting simple linear regression to traing set
from sklearn.linear_model import LinearRegression                  
regressor=LinearRegression()
regressor.fit(xtrain, ytrain)

#prediction of test result
ypred=regressor.predict(xtest)

#visulising test result
plt.scatter(xtest, ytest, color='red')
plt.plot(xtest,regressor.predict(xtest), color='blue')
plt.title("experience vs salary (test data)")
plt.xlabel("year of experience")
plt.ylabel("salary")
plt.show() 

#visulising train result
plt.scatter(xtrain, ytrain, color='red')
plt.plot(xtrain,regressor.predict(xtrain), color='blue')
plt.title("experience vs salary (train data)")
plt.xlabel("year of experience")
plt.ylabel("salary")
plt.show() 