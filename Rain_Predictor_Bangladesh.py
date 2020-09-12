#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Reading data file
data = pd.read_csv('Temp_and_rain.csv')

#Data Visualisation
data.head()
sns.pairplot(data)
data.describe()

#Taking a subset of rows for Monsoon season
ndata = data[data["Month"].isin([6,7,8,9])]
print(ndata)

#Training and fitting data in a model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = ndata[['tem', 'Month', 'Year']]
y= ndata['rain']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=200)

lm.fit(X_train,y_train) 
predicted = lm.predict(X_test)

print(predicted)

#Checking fit by means of a scatter plot
plt.scatter(y_test,lm.predict(X_test)) #x,y
plt.xlabel('Y Test')
plt.ylabel('Y Predicted')

#----------------------------------------------------------------
#X = ndata[['tem', 'Month', 'Year']]
#months - [6,9]

#Example:
Xnew = [[30,7,2020]]
prediction = lm.predict(Xnew)
print(prediction)
#[251.48727646] answer to example
