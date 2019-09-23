import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

#Read the data
data = pd.read_csv("network_backup_dataset.csv")


"""==========================================Util methods==========================================================="""

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def scalar_encode_day(days):
    day_to_num = dict(zip(list(calendar.day_name), range(1, 8)))
    return [day_to_num[day] for day in days]

def scalar_encode_underscored(file_names):
    return [int(name.split('_')[-1]) for name in file_names]

def mse(predicted, actual):
    return mean_squared_error(predicted, actual)


def linear_regression(X, y, plot=True, model=LinearRegression()):
    kf = KFold(n_splits=10, shuffle=True)

    cv_train_rmse, cv_test_rmse = [], []
    min_test_rmse = float("inf")

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lm = model
        lm.fit(X_train, y_train)
        train_rmse = mse(lm.predict(X_train), y_train)
        test_rmse = mse(lm.predict(X_test), y_test)

        if test_rmse < min_test_rmse:
            min_test_rmse = test_rmse
            best_model = lm

        cv_train_rmse.append(train_rmse)
        cv_test_rmse.append(test_rmse)

    avg_train_rmse = np.sqrt(np.mean(cv_train_rmse))
    avg_test_rmse = np.sqrt(np.mean(cv_test_rmse))

    print("Coefficients are : ", best_model.coef_)
    best_model_y_predict = best_model.predict(X)

    if plot:

        best_model_y_predict = best_model_y_predict.reshape(y.shape)

        plt.figure(figsize=(8, 5))
        plt.scatter(y, best_model_y_predict, s=5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title("Fitted values vs True values")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.scatter(best_model_y_predict, y - best_model_y_predict, s=5)
        plt.xlabel("Fitted values")
        plt.ylabel("Residual values")
        plt.title("Residual values vs Fitted values")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(y)), y, c='g', marker='x', label='True values')
        plt.scatter(range(len(y)), best_model_y_predict, c='b', marker='o', label='Predicted Values')
        plt.title("Fitted values and True values")
        plt.legend(loc='upper left')
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(y)), best_model_y_predict, c='g', s=5, zorder=2, label='Fitted values')
        plt.scatter(range(len(y)), np.subtract(y, best_model_y_predict), c='b', s=5, zorder=1, label='Residual Values')
        plt.title("Fitted values and Residual values")
        plt.legend(loc='upper left')
        plt.yscale('log')
        plt.show()

    return avg_train_rmse, avg_test_rmse


def f_regr(X, Y):
   return f_regression(X,Y,center = False)

def mutual_info_regr(X, Y):
    return mutual_info_regression(X, Y)


def scalar_encode(data, input_features, output_col_name):
    X = data[input_features].values

    y = data[output_col_name].values

    X[:, 1] = scalar_encode_day(X[:, 1])
    X[:, 3] = scalar_encode_underscored(X[:, 3])
    X[:, 4] = scalar_encode_underscored(X[:, 4])

    return X, y


"""===================================================Question 1====================================================="""

data['day_number'] = (data['Week #']-1)*7+ scalar_encode_day(data['Day of Week'])
data['Workflow_ID'] = scalar_encode_underscored(data['Work-Flow-ID'])

#Question 1a)
data_20 = data[data['day_number']<=20]

plt.figure(figsize=(8,5))
plt.scatter(x=data_20['day_number'],y=data_20['Size of Backup (GB)'],c=data_20['Workflow_ID'], s = 8)
plt.xlabel('Day number')
plt.ylabel('Backup size (GB)')
plt.show()

#Question 1b)
plt.figure(figsize=(8,5))
plt.scatter(x=data['day_number'],y=data['Size of Backup (GB)'],c=data['Workflow_ID'], s = 8)
plt.xlabel('Day number')
plt.ylabel('Backup size (GB)')
plt.show()


"""===================================================Question 2a (i)================================================"""

print("Fit linear regression model after scalar encoding")
input_features = ['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', 'Work-Flow-ID', 'File Name']
output_col_name = ['Size of Backup (GB)']
X, y = scalar_encode(data, input_features, output_col_name)
linear_regression(X, y)

"""===================================================Question 2a (ii)==============================================="""

print("Fit linear regression model after data preprocessing - Standardizing")
scaler = StandardScaler()
std_input = np.copy(X)
scaler.fit(std_input)
linear_regression(scaler.transform(std_input), y)

"""==================================================Question 2a (iii)==============================================="""

print("Linear Regression Model after Feature Selection")
most_imp_var_num = 3

print("f_regression")
X_f_regr = SelectKBest(score_func = f_regr, k = most_imp_var_num).fit_transform(X, y)
linear_regression(X_f_regr, y)

print("Mutual Information Regression")
X_mutualinfo_regr = SelectKBest(score_func = mutual_info_regr, k = most_imp_var_num).fit_transform(X, y)
linear_regression(X_mutualinfo_regr, y)


"""===================================================Question 2a(iv)================================================"""

feature_range = range(5)
feature_powerset = list(powerset(feature_range))

i = 0
avg_train_rmse, avg_test_rmse = [], []

for combination in feature_powerset:
    X_copy = np.copy(X)
    print("\n", i, " :  One-Hot encoded features :", np.array(input_features)[list(combination)])
    enc = OneHotEncoder(categorical_features=list(combination))
    onehot_encoded = enc.fit_transform(X_copy)

    if type(onehot_encoded) != np.ndarray:
        onehot_encoded = onehot_encoded.toarray()

    train_rmse, test_rmse = linear_regression(onehot_encoded, y, plot=False)
    avg_train_rmse.append(train_rmse)
    avg_test_rmse.append(test_rmse)

    i = i + 1

X_copy = np.copy(X)
enc = OneHotEncoder(categorical_features = [1, 2, 3, 4])
onehot_encoded = enc.fit_transform(X_copy)

if type(onehot_encoded) != np.ndarray:
    onehot_encoded = onehot_encoded.toarray()

train_rmse, test_rmse = linear_regression(onehot_encoded, y, plot = True)

x_range = range(32)
plt.figure(figsize=(8,5))
plt.plot(x_range, avg_train_rmse)
plt.ylabel("Average Train RMSE")
plt.title("Average Train RMSE for different one-hot combinations")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(x_range, avg_test_rmse)
plt.ylabel("Average Test RMSE")
plt.title("Average Test RMSE for different one-hot combinations")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(x_range, avg_test_rmse)
plt.yscale('log')
plt.ylabel("Average Test RMSE")
plt.title("Average Test RMSE for different one-hot combinations")
plt.show()

print(avg_train_rmse)
print(avg_test_rmse)
print("Minimum Train RMSE " , min(avg_train_rmse), " for combination " , np.argmin(avg_train_rmse))
print("Minimum Test RMSE " ,  min(avg_test_rmse), " for combination " ,  np.argmin(avg_test_rmse))


"""===================================================Question 2a(v)================================================="""

alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5,10]

def regularization_optimization(model, alpha_range=alpha_values):
    alpha_rmse = []

    for alpha in alpha_range:

        # print('Testing for Regularization strength = ', alpha)
        avg_test_rmse = []
        for combination in feature_powerset:
            X_copy = np.copy(X)
            enc = OneHotEncoder(categorical_features=list(combination))
            onehot_encoded = enc.fit_transform(X_copy)

            if type(onehot_encoded) != np.ndarray:
                onehot_encoded = onehot_encoded.toarray()

            _, test_rmse = linear_regression(onehot_encoded, y, plot=False, model=model(alpha=alpha))
            avg_test_rmse.append(test_rmse)

        alpha_rmse.append(avg_test_rmse)

    return alpha_rmse


regularizations = ["Ridge", "Lasso", "ElasticNet"]


for reg in regularizations:

    print("Finding the best parameters for ", reg)
    plt.figure(figsize=(8, 5))
    plt.xlabel("One-hot encoded Model")
    plt.ylabel("Test RMSE")
    plt.title("Test RMSE for One-hot encoded models for " + reg)

    # alpha_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5,10]
    alpha_range = alpha_values
    alpha_rmse = regularization_optimization(model=eval(reg), alpha_range=alpha_range)

    for alpha in range(len(alpha_range)):
        print(alpha, min(alpha_rmse[alpha]), np.argmin(alpha_rmse[alpha]))
        plt.plot(x_range, alpha_rmse[alpha], label=alpha_range[alpha])

    plt.legend(title="Regularization strength")
    plt.show()



X_copy = np.copy(X)
enc = OneHotEncoder(categorical_features = [1, 2, 3, 4])
onehot_encoded = enc.fit_transform(X_copy)

if type(onehot_encoded) != np.ndarray:
    onehot_encoded = onehot_encoded.toarray()

train_rmse, test_rmse = linear_regression(onehot_encoded, y, model = Lasso(alpha = 0.001))
print(train_rmse,test_rmse)

train_rmse, test_rmse = linear_regression(onehot_encoded, y, model = ElasticNet(alpha = 0.001))
print(train_rmse,test_rmse)

train_rmse, test_rmse = linear_regression(onehot_encoded, y, model = Ridge(alpha = 0.001))
print(train_rmse,test_rmse)


"""=================================================================================================================="""