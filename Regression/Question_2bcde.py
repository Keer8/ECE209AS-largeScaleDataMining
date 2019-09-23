# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 07:45:35 2018

@author: Amruth
"""
#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor

###############################################################################
#Question 1
###############################################################################
#Loading data
data = pd.read_csv('network_backup_dataset.csv')

#Finding the day number
day_map = {'Monday':1, 'Tuesday':2, 'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
day_encoded = [day_map[i] for i in data['Day of Week']]
data['day'] = day_encoded
data['day_number'] = (data['Week #']-1)*7+day_encoded
data['Workflow_ID'] = [int(i[10]) for i in data['Work-Flow-ID']]

#Plotting question 1a
data_20 = data[data['day_number']<=20]
plt.scatter(x=data_20['day_number'],y=data_20['Size of Backup (GB)'],c=data_20['Workflow_ID'])
plt.xlabel('Day number')
plt.ylabel('Backup size (GB)')
plt.legend()
plt.show()

#Plotting question 1b
plt.scatter(x=data['day_number'],y=data['Size of Backup (GB)'],c=data['Workflow_ID'])
plt.xlabel('Day number')
plt.ylabel('Backup size (GB)')
plt.legend()
plt.show()

###############################################################################
#Question 2ai
###############################################################################

#Getting the numerical value as a column 
data['File_number'] = [int(i[5:]) for i in data['File Name']]
scalar_data = data[['Week #','day','Backup Start Time - Hour of Day','Workflow_ID',
                    'File_number','Size of Backup (GB)']]
scalar_data_X = scalar_data[['Week #','day','Backup Start Time - Hour of Day',
'Workflow_ID','File_number']]
scalar_data_y = scalar_data[['Size of Backup (GB)']]                     
                      
#K Fold Cross validation as a function
def kfold_cv(X,y,return_errors=False,regression='Linear',features=5,trees=20,depth=4,k=5,
             return_model=False):
    train_error_k, test_error_k = [], []
    kf = KFold(n_splits=5,shuffle=False)
    best_mse=1e8
    for trainset, testset in kf.split(X):
        if regression == 'KNN':
            #print('Using KNN Regression')
            regressor = KNeighborsRegressor(n_neighbors=k)
        elif regression == 'RF':
            #print('Using random forest regression')
            regressor = RandomForestRegressor(n_estimators=trees,max_depth=depth,
                                              max_features=features,
                                              bootstrap=True)
        else:
            regressor = LinearRegression()
        regressor.fit(X=X.iloc[trainset],y=y.iloc[trainset].values.ravel())
        y_pred_train = regressor.predict(X.iloc[trainset])
        train_mse = mean_squared_error(y.iloc[trainset],y_pred_train) 
        train_error_k.append(train_mse)
        y_pred_test = regressor.predict(X.iloc[testset])
        test_mse = mean_squared_error(y.iloc[testset],y_pred_test)
        test_error_k.append(test_mse)
        if test_mse < best_mse:
            best_model = regressor
    training_error = np.sqrt(np.mean(train_error_k))
    test_error = np.sqrt(np.mean(test_error_k))    
    print('The training error is', training_error)    
    print('The testing error is', test_error)    
    if return_errors:
        return training_error,test_error
    if return_model:
        return best_model
    
def show_plots(X,y,regressor):
    y = y.values.ravel()
    regressor.fit(X=X,y=y) 
    y_pred = regressor.predict(X)
    plt.scatter(range(len(X)),y_pred,c='red',label='Fitted value',edgecolors='none',zorder=2)
    plt.scatter(range(len(X)),y,c='blue',label='True value',edgecolors='none',zorder=1)
    #plt.ylabel('Fitted value')
    #plt.xlabel('True value')
    plt.title('Fitted value vs true value for the model')
    plt.legend()
    #plt.xlim(-0.1,1.1)
    #plt.ylim(-0.1,1.1)
    plt.show()
    
    plt.scatter(range(len(X)),y_pred,c='red',label='Fitted value',edgecolors='none',zorder=1)
    plt.scatter(range(len(X)),y-y_pred,c='blue',label='Residual',edgecolors='none',zorder=2)
    #plt.xlabel('Fitted values')
    #plt.ylabel('Residuals')
    plt.title('Residuals vs fitted values for the model')
    plt.legend()
    #plt.xlim(-0.1,1.1)
    #plt.ylim(-0.1,1.1)
    plt.show()  
    
#Question 2ai        
kfold_cv(scalar_data_X,scalar_data_y)
show_plots(scalar_data_X,scalar_data_y,regressor=LinearRegression())
    
###############################################################################
#Question 2a(ii)
###############################################################################

sc = StandardScaler()
scaled_X = pd.DataFrame(sc.fit_transform(scalar_data_X))
kfold_cv(scaled_X,scalar_data_y)
show_plots(scaled_X,scalar_data_y,regressor=LinearRegression())

###############################################################################
#Question 2a(iii)
###############################################################################

F,_ = f_regression(scalar_data_X,scalar_data_y.values.ravel())
top_features = np.argsort(F)[-3:]
print('The top features are', scalar_data_X.columns.values[top_features])
f_top_X = scalar_data_X[scalar_data_X.columns.values[top_features]]
kfold_cv(f_top_X,scalar_data_y)
show_plots(f_top_X,scalar_data_y,regressor=LinearRegression())

mi = mutual_info_regression(scalar_data_X,scalar_data_y.values.ravel())
top_features = np.argsort(mi)[-3:]
print('The top features are', scalar_data_X.columns.values[top_features])
f_top_X = scalar_data_X[scalar_data_X.columns.values[top_features]]
kfold_cv(f_top_X,scalar_data_y)
show_plots(f_top_X,scalar_data_y,regressor=LinearRegression())

###############################################################################
#Question 2a(iv)
###############################################################################

onehotencoder = OneHotEncoder(categorical_features = 'all')
onehot_week = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['Week #'].reshape(-1,1)).toarray()[:,1:])
onehot_day = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['day'].reshape(-1,1)).toarray()[:,1:])
onehot_time = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['Backup Start Time - Hour of Day'].reshape(-1,1)).toarray()[:,1:])
onehot_workflow = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['Workflow_ID'].reshape(-1,1)).toarray()[:,1:])
onehot_file = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['File_number'].reshape(-1,1)).toarray()[:,1:])

scalar_week = pd.DataFrame(scalar_data_X['Week #'])
scalar_day = pd.DataFrame(scalar_data_X['day'])
scalar_time = pd.DataFrame(scalar_data_X['Backup Start Time - Hour of Day'])
scalar_workflow = pd.DataFrame(scalar_data_X['Workflow_ID'])
scalar_file = pd.DataFrame(scalar_data_X['File_number'])

d = {0:[scalar_week,onehot_week],1:[scalar_day,onehot_day],2:[scalar_time,onehot_time],
        3:[scalar_workflow,onehot_workflow],4:[scalar_file,onehot_file]}

train_err,test_err = [], []
for number in range(32):
    num = np.binary_repr(number, width=5)
    frames = []
    for i in range(len(num)):
        if int(num[i])==0:
            frames.append(d[i][0])
        else:
            frames.append(d[i][1])            
    result = pd.concat(frames,axis=1)
    train,test = kfold_cv(result,scalar_data_y,return_errors=True)
    train_err.append(train)
    test_err.append(test)
    
plt.plot(range(1,33),test_err,c='blue',label='Testing error')
plt.legend()
plt.xlabel('Combination number')
plt.ylabel('Root mean square error')
plt.title('Testing error for different combinations')
plt.show()    

plt.plot(range(1,33),train_err,c='red',label='Training error')
plt.legend()
plt.xlabel('Combination number')
plt.ylabel('Root mean square error')
plt.title('Training error for different combinations')
plt.show()    


###############################################################################
#Question 2d(i),(ii)
###############################################################################

num_workflow = len(set(scalar_data['Workflow_ID']))
for i in range(num_workflow):
    subset_data = scalar_data[scalar_data['Workflow_ID']==i]
    y = subset_data[[-1]]
    X = subset_data[['Week #', 'day', 'Backup Start Time - Hour of Day','File_number']]
    print('The workflow ID number is', i)
    best_model = kfold_cv(X,y,return_model=True)
    show_plots(X,y,best_model)
    
    
for i in range(num_workflow):
    train_err = []
    test_err = []
    subset_data = scalar_data[scalar_data['Workflow_ID']==i]
    y = subset_data[[-1]]
    X = subset_data[['Week #', 'day', 'Backup Start Time - Hour of Day','File_number']]
    for degree in range(2,10):
        poly_reg = PolynomialFeatures(degree = degree)
        X_poly = pd.DataFrame(poly_reg.fit_transform(X))
        train,test=kfold_cv(X_poly,y,return_errors=True)
        train_err.append(train)
        test_err.append(test)
        
    plt.plot(range(2,10),train_err,c='red',label='Training error')
    plt.plot(range(2,10),test_err,c='blue',label='Testing error')
    plt.legend()
    plt.xlabel('Polynomial degree')
    plt.ylabel('Root mean square error')
    plt.title('Error for polynomial degrees for workflow id = %d'%i)
    plt.show()    
    
###############################################################################
#Question 2e
###############################################################################
min_error = 1e5
train_err,test_err = [], []
for k in range(1,11):
    train,test = kfold_cv(scaled_X,scalar_data_y,regression='KNN',k=k,return_errors=True)
    train_err.append(train)
    test_err.append(test)
    if test<min_error:
        min_error = test
        best_k = k
plt.plot(range(1,11),test_err,label='Test error')
plt.ylabel('Test RMSE')
plt.xlabel('Number of neighbors')
plt.title('Test error for different number of neighbors')
plt.legend()
plt.show()  

best_model=kfold_cv(scaled_X,scalar_data_y,regression='KNN',k=best_k,return_model=True)  
show_plots(scaled_X,scalar_data_y,best_model)

###############################################################################
#Question 2c
###############################################################################

onehots = [onehot_week,onehot_day,onehot_time,onehot_workflow,onehot_file]
onehot_data = pd.concat(onehots,axis=1)
hidden_size_range = range(10,510,10)
test_error_act = []
best_err = 1e7
for activation in ('relu','logistic','tanh'):
    test_error_h = []
    for hidden_size in hidden_size_range:
        print('Hidden size is', hidden_size)
        test_error_k = []
        kf = KFold(n_splits=10,shuffle=True)
        for trainset, testset in kf.split(onehot_data):
            regressor = MLPRegressor(hidden_layer_sizes=(hidden_size,),activation=activation)
            regressor.fit(X=onehot_data.iloc[trainset],y=scalar_data_y.iloc[trainset])
            y_pred_test = regressor.predict(onehot_data.iloc[testset])
            test_mse = mean_squared_error(scalar_data_y.iloc[testset],y_pred_test)
            test_error_k.append(test_mse)  
        test_rmse = np.sqrt(np.mean(test_error_k))
        if test_rmse<best_err:
            best_err = test_rmse
            best_model = regressor
        test_error_h.append(test_rmse)  
    test_error_act.append(test_error_h)
plt.plot(hidden_size_range,test_error_act[0],c='red',label='Relu activation')
plt.plot(hidden_size_range,test_error_act[1],c='blue',label='Logistic activation')
plt.plot(hidden_size_range,test_error_act[2],c='green',label='Tanh activation')
plt.legend(loc=5)
plt.xlabel('Hidden layer size')
plt.ylabel('Test error')
plt.title('Test error for different hidden layer size')
plt.show()
print('The best model is:',best_model)

###############################################################################
#Question 2b(i)
###############################################################################

kfold_cv(scalar_data_X,scalar_data_y,regression='RF')
show_plots(scalar_data_X,scalar_data_y,regressor=RandomForestRegressor(n_estimators=20,max_depth =4,oob_score=True,
                                                                       max_features=5,bootstrap=True))

def oob(X,y,features=5,trees=20,depth=4):
    regressor = RandomForestRegressor(n_estimators=trees,max_depth =depth,max_features=features,
                                      oob_score=True, bootstrap=True)
    regressor.fit(X,y.values.ravel())
    return (1-regressor.oob_score_)
    
print(oob(scalar_data_X,scalar_data_y))

###############################################################################
#Question 2b(ii)
###############################################################################
tree_range = range(1,201)
oob_feats,testerr_feats = [],[]
for num_feats in range(1,6):
    oob_trees,testerr_trees = [],[]
    for num_trees in tree_range:
        oob_trees.append(oob(scalar_data_X,scalar_data_y,features=num_feats,trees=num_trees))
        _,test_err = kfold_cv(scalar_data_X,scalar_data_y,return_errors=True,regression='RF',
                              features=num_feats,trees=num_trees)
        testerr_trees.append(test_err)
    oob_feats.append(oob_trees)
    testerr_feats.append(testerr_trees)

for i in range(5):
    plt.plot(tree_range,testerr_feats[i],label='Max features = '+str(i+1))
plt.legend()
plt.xlabel('Number of trees')
plt.ylabel('Test error')
plt.title('Test error for different tree size')
plt.show() 
   
for i in range(5):    
    plt.plot(tree_range,oob_feats[i],label='Max features = '+str(i+1))
plt.xlabel('Number of trees')
plt.ylabel('Out of bag error')
plt.legend()
plt.title('Out of bag error for different tree size')
plt.show()    

###############################################################################
#Question 2b(iii)
###############################################################################
tree_range = range(1,201)
oob_depth,testerr_depth = [],[]
for num_depth in range(1,6):
    oob_trees,testerr_trees = [],[]
    print('depth is', num_depth)
    for num_trees in tree_range:
        oob_trees.append(oob(scalar_data_X,scalar_data_y,depth=num_depth,trees=num_trees))
        _,test_err = kfold_cv(scalar_data_X,scalar_data_y,return_errors=True,regression='RF',
                              depth=num_depth,trees=num_trees)
        testerr_trees.append(test_err)
    oob_depth.append(oob_trees)
    testerr_depth.append(testerr_trees)

for i in range(5):
    plt.plot(tree_range,testerr_depth[i],label='Max depth = '+str(i+1))
plt.legend()
plt.xlabel('Number of trees')
plt.ylabel('Test error')
plt.title('Test error for different tree size')
plt.show() 
   
for i in range(5):    
    plt.plot(tree_range,oob_depth[i],label='Max depth = '+str(i+1))
plt.xlabel('Number of trees')
plt.ylabel('Out of bag error')
plt.legend()
plt.title('Out of bag error for different tree size')
plt.show()

###############################################################################
#Question 2b(v)
###############################################################################

dec_tree = DecisionTreeRegressor(max_depth=4,max_features=3)
dec_tree.fit(scalar_data_X,scalar_data_y.values.ravel())
export_graphviz(dec_tree,out_file='tree2.dot') 

###############################################################################
#Question 2b(iv)
###############################################################################

regressor = RandomForestRegressor(n_estimators=70,max_depth =5,max_features=3,
                                  oob_score=True, bootstrap=True)
regressor.fit(scalar_data_X,scalar_data_y.values.ravel())
importances = regressor.feature_importances_
print(importances)


