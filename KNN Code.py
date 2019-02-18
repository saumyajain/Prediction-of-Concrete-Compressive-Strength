# Importing the libraries
import numpy as np
from sklearn import cross_validation, neighbors
import pandas as pd

#Importing Data
data1=pd.read_excel("Concrete_Data.xls")
print(data1.head())
data1.columns.values
data1.columns = ['cement','blast_furnace','fly_ash','water', 'superplasticizer','coarse_agg','fine_agg','age','strength']

#Assigning values of strength to Y
Y=np.array(data1[data1.columns[-1]])

#Creating labels for 5 classes
a=1030*[1]
for i in range(0,1030):
    if Y[i] < 20:
        a[i]= "C1"
    elif Y[i] < 40:
        a[i]= "C2"
    elif Y[i] < 60:
        a[i]= "C3"    
    elif Y[i] < 80:
        a[i]= "C4"    
    else:
        a[i]= "C5"       

data1['label']=a   
del i
del a
     
#Assigning X as independent and Y as dependent variables
X=np.array(data1.drop(data1.columns[-2:],axis=1))
Y=np.array(data1[data1.columns[-1]]).reshape((data1.shape[0]),1)

#Randomly splitting data into training and testing data
X_train, X_test,Y_train, Y_test=cross_validation.train_test_split(X,Y,test_size=0.1)

#Defining and fitting KNN model on training dataset
clf= neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,Y_train)

#Printing accuracy of model on training and testing data
print(clf.score(X_train,Y_train))
print(clf.score(X_test,Y_test))

#predicting classes for testing data and compairing with actual values
print(clf.predict(X_test))
Y_test

#corelation between all variables
relation=data1.corr()
relation