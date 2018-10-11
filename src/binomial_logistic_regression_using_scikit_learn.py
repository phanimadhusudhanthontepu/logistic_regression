# -*- coding: utf-8 -*-

# Imports
#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the Dataset
# Iris Dataset
dataset = pd.read_csv('..//data//titanic_dataset//iris.csv')
x_labels = ['sepal length', 'sepal width', 'petal length', 'petal width']
y_labels = ['iris']

# Understanding the data
if dataset.isnull().values.any():
    print('There are null in the dataset')
else:
    print('There are no nulls in the dataset') # In case nulls are there much more preprocessing is required by replacing nulls with appropriate values

#dataset.info() # To Know the columns in the dataset and types of values and number of values
print(dataset.describe()) # To know min,max,count standard diviation ,varience in each column which would tell us if there is any outliers,normalization or standadization required.
print(dataset.head()) # To view first five columns of the dataset

# Outlier check with the help of histogram
dataset.hist(column = x_labels,bins=20, figsize=(10,5))

# Visualizing the dataset
sns.pairplot(dataset, hue=y_labels[0], size=2.5,markers=["o", "s", "D"])

# Extracting dependent and Independent varibles
X = dataset[x_labels].values
y = dataset[y_labels].values
y = y.ravel() # we would be requiring 1-D array for processing in further code

# Test-Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y_train = le.transform(y_train)
y_test = le.transform(y_test)


# Building and Training the classifier class
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train) # Trainign the classifier

# Predicting the classes
y_pred = clf.predict(X_test)

# Creating Confusion matrix 
y_pred = le.inverse_transform(y_pred)
y_test = le.inverse_transform(y_test)

from sklearn.metrics import confusion_matrix
classes = list(set(y))
cm = confusion_matrix(y_test,y_pred,labels=classes)
print(cm)

# Visualizing confusion matrix
df_cm = pd.DataFrame(cm,index = [i for i in classes],columns = [i for i in classes])
plt.figure(figsize = (10,7))
cm_plot = sns.heatmap(df_cm, annot=True)
cm_plot.set(xlabel='Predicted', ylabel='Actual7')
plt.show()
