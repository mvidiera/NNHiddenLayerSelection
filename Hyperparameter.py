# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importing the dataset. this has info which has customer's bank info and it predicts whether customer quits bank or not 

dataset = pd.read_csv('/Users/melissavidiera/Documents/Deep Learning/Hidden-Layers-Neurons-master/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables. geo and gender is binary data. Converting into 0 and 1
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames. concatenating dummycolmns geo and gender to X training data 
X=pd.concat([X,geography,gender], axis=1)

## Drop Unnecessary columns. Dropping original columns of geo and gender
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Perform Hyperparameter Optimization

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid



def create_model(layers, activation): #create sequential model where input is layers and activation 
    model = Sequential()# create sequential model
    for i, nodes in enumerate(layers):
        if i==0: #first node is input node where I need to load training data as input data. using dense create node of "nodes" ip which has been passed as arg
            model.add(Dense(nodes,input_dim=X_train.shape[1])) #shape of X_train is 8k, 11. X_train.shape[1] returns shape of 11. 
            model.add(Activation(activation)) #use actvation func passed as arg
            model.add(Dropout(0.3)) #use 30% as dropout ratio
        else:
            model.add(Dense(nodes)) #for rest of hidden layers which are not input layers, create nodes using dense() and nodes number of neuron
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    #for last layer output: 1 node as it is binary classification problem. activation used is sigmoid        
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    #complile the mode. calculare loss, using adam optimizer reduce loss 
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy']) #metrics func is used to judge the performance
    return model  #accuracy metrics creates 2 local var: total and count. calculates how many y_pred matches with y_true: total. then it will be divided by count.
    

# 1st parameter: build- here I am passing my wriiten function. verbose
model = KerasClassifier(build_fn=create_model, verbose=1)
#verbose is used to show how ouput works if 0 then silent. if 1 then it shows generating output in bar graph type 
# research suggests, ReLU works well with he_uniform and glorot uniform for op node


#how to decide lyers and act func
#in layers I have a dictionary, in 1st iteration consider just 1 hidden layer of 20 neurons. 2nd iteration 2 hidden layer 40 and 20 neurons.
# 3rd iteration, 3 hidden layer od 45, 30, 15 neurons in each layer. 
#these dictionary of layers and activation are passed as arguments for my function create_model
layers = [[20], [40, 20], [45, 30, 15]] 
activations = ['sigmoid', 'relu'] # for 1st iteration use sigmoid, second ReLu, whichever is better it will be selected 
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30]) 


grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5) 
#crss validation cv: to avoid overfitting even after train test split. training data will be divided into 5 parts if cv is 5. 5X5
#n each iteration 5, 1 part is left and 4  other parts are considered as training data. 5 iterations. this otput is validated with the 1/5th part of each iteration which was left 
grid_result = grid.fit(X_train, y_train)

#Error: To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Best result of the model
print(grid_result.best_score_,grid_result.best_params_)
# activation is relu, batch_size 128, epoch 30, layers: 3 hidden layers 40,30,15

pred_y= grid.predict(X_test) #predict using x_test data. 
y_pred= (pred_y> 0.5) # in pred_y the ones which are more than 0.5, store in y_pred. in this case all are

from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_pred, y_test) #check cm in variable explorer 
score= accuracy_score(y_pred,y_test) # check score in variable explorer : Score = 86%















