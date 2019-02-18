# Importing the libraries
import numpy as np
import pandas as pd

#Importing Data
data1=pd.read_excel("Concrete_Data.xls")
data1.columns = ['cement','blast_furnace','fly_ash','water', 'superplasticizer','coarse_agg','fine_agg','age','strength']

#Assigning values of strength to Y
Y=np.array(data1[data1.columns[-1]])

#Creating labels for 5 classes
a=1030*[1]

for i in range(0,1030):
    if Y[i] < 20:
        a[i]= 0
    elif Y[i] < 40:
        a[i]= 1
    elif Y[i] < 60:
        a[i]= 2       
    elif Y[i] < 80:
        a[i]= 3    
    else:
        a[i]= 4
      
data1['label']=a
del i
del a

#Assigning X as independent and Y as dependent variables
X=np.array(data1.drop(data1.columns[-2:],axis=1))
Y=np.array(data1[data1.columns[-1]]).reshape((data1.shape[0]),1)

#Normalizing dataset
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)

#Randomly splitting data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Defining ANN model with 3 layers
classifier = Sequential([Dense(16,input_dim=8, activation = 'relu'),Dense(32,activation='relu'),Dense(5,activation='softmax')])

# Compiling Neural Network
classifier.compile(Adam(lr=0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting our model 
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 1000, verbose=2)

#Evaluating [Loss,Accuracy] for training and testing dataset
classifier.evaluate(X_train,Y_train, batch_size = 10, verbose=1)
classifier.evaluate(X_test,Y_test, batch_size = 10, verbose=1)

# Predicting the Test set results
Y_pred_train = classifier.predict_classes(X_train, batch_size = 10)
Y_pred_test = classifier.predict_classes(X_test, batch_size = 10)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(Y_train, Y_pred_train)
cm_test = confusion_matrix(Y_test, Y_pred_test)

cm_train
cm_test