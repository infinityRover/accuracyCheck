# Python program to compute accuracy score using the function compute_accuracy  
# Importing the required libraries
  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.datasets import load_iris  
  
# Loading the dataset  
X, Y = load_iris(return_X_y = True)  
  
# Splitting the dataset in training and test data  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)  
  
# Training the model using the Support Vector Classification class of sklearn  
svc = SVC()  
svc.fit(X_train, Y_train)  
  
# Computing the accuracy score of the model  
Y_pred = svc.predict(X_test)  
score = compute_accuracy(Y_test, Y_pred)  
print(score)