#import all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#import the dataset
dataset = pd.read_csv(r"C:\Users\dell\Documents\DATA SCIENCE,AI & ML\13th\SIMPLE LINEAR REGRESSION\Salary_Data.csv")



#split the dataset into D.V and I.V as "y" and "x"
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values



#split the dataset into train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state=0)



#applying the Simple Linear Regression Algorithm
from sklearn.linear_model import LinearRegression            #package calling
regressor = LinearRegression()                               #create a variable named "regressor" to hold the algorithm
regressor.fit(x_train,y_train)                               #build the model using x_train and y_train
y_pred = regressor.predict(x_test)                           #pass x_test to the model and model will predict y_pred table



#plotting the scatter plot for the training set to know the model performance
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Experience(Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



#plotting the scatter plot for the testing set to know the model performance
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Experience(Testing Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



#performance of the training data
bias = regressor.score(x_train,y_train)
bias



#performance of the testing data
variance = regressor.score(x_test,y_test)
variance
