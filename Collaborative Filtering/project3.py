# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:07:42 2018

@author: Amruth
"""
#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection.validation import cross_validate
from surprise import accuracy
from surprise.model_selection import KFold
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import NMF
from sklearn.metrics import mean_squared_error

ratings = pd.read_csv('ratings.csv')
movie = pd.read_csv('movies.csv')
df = pd.DataFrame({'itemID': list(ratings.movieId), 'userID': list(ratings.userId), 'rating': list(ratings.rating)})
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

#Question 1
num_movies = len(set(ratings['movieId']))
num_users = len(set(ratings['userId']))
sparsity = len(ratings) / (num_movies * num_users)

#Question 2
plt.hist(ratings['rating'],bins = np.arange(0.25,5.5,0.5),color = '#58CCF2')

#Question 3
c1 = Counter(ratings['movieId'])
plt.bar(np.arange(num_movies),sorted(c1.values(),reverse = True))
plt.xticks([], [])
plt.title('Distribution of ratings among movies')
plt.ylabel('Number of ratings')
plt.xlabel('Movies')

#Question 4
c2 = Counter(ratings['userId'])
plt.bar(np.arange(num_users),sorted(c2.values(),reverse = True))
plt.xticks([], [])
plt.title('Distribution of ratings among users')
plt.ylabel('Number of ratings')
plt.xlabel('Users')

#Question 6
var = ratings.groupby('movieId')['rating'].var().fillna(0).tolist()
plt.hist(var, bins = np.arange(0,11,0.5))
plt.xlabel('Variance of ratings')
plt.ylabel('Number of movies')
plt.title('Distribution of variance of ratings')

#Question 10
k_range = range(2,100,2)
avg_rmse, avg_mae = [], []
for k in k_range:
    algo = KNNWithMeans(k=k, sim_options = {'name':'pearson'}) 
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=False)
    avg_rmse.append(np.mean(cv_results['test_rmse']))
    avg_mae.append(np.mean(cv_results['test_mae']))
    
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.plot(k_range, avg_mae, label = "Average MAE")
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()


#Functions for Question 12,13,14
def popular_trim(data):
    print("Popular trimming")
    movie_id_counter = Counter([val[1] for val in data])
    popular_trimmed_data = [val for val in data if movie_id_counter[val[1]] > 2]
    return popular_trimmed_data
    
def unpopular_trim(data):
    print("Unpopular trimming")
    movie_id_counter = Counter([val[1] for val in data])
    popular_trimmed_data = [val for val in data if movie_id_counter[val[1]] <= 2]
    return popular_trimmed_data

def high_var_trim(data):
    print("High variance trimming")
    movie_rating_map = defaultdict(list)
    for val in data:
        movie_rating_map[val[1]].append(val[2])
    high_var_data = [val for val in data if len(movie_rating_map[val[1]]) >= 5 and np.var(movie_rating_map[val[1]]) >= 2.0]
    return high_var_data

#Question 12
k_range = range(2,100,2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = KNNWithMeans(k=k, sim_options = {'name':'pearson'}) 
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(popular_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))
        
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' %(np.min(avg_rmse),np.argmin(avg_rmse)))

#Question 13
k_range = range(2,100,2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = KNNWithMeans(k=k, sim_options = {'name':'pearson'}) 
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(unpopular_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))
        
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' %(np.min(avg_rmse),np.argmin(avg_rmse)))

#Question 14
k_range = range(2,100,2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = KNNWithMeans(k=k, sim_options = {'name':'pearson'}) 
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(high_var_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))
        
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' %(np.min(avg_rmse),np.argmin(avg_rmse)))

#Question 15
train_set, test_set = train_test_split(data, test_size=0.1, random_state=0)
thresholds = [2.5,3,3.5,4]
algo = KNNWithMeans(k=20, sim_options = {'name':'pearson'}) 
algo.fit(train_set)
predictions = algo.test(test_set)
pred_est = np.array([i.est for i in predictions])
actual_ratings = np.array([i.r_ui for i in predictions])
for threshold in thresholds:
    y_score = pred_est
    y_true = actual_ratings>=threshold
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='Threshold = %0.1f, AUC = %0.4f' % (threshold,roc_auc))
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate')
    plt.legend(loc ='lower right')
    plt.title('Receiver Operating Characteristics (ROC) for threshold = %0.1f' %threshold)
    plt.show()
    
#Question 30
avg_rating = df.groupby(['userID'])['rating'].mean().tolist()
def naive_prediction(dataset):
    predictions = [avg_rating[dataset[i][0]-1] for i in range(len(dataset))]
    return predictions

k_rmse = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    y_pred = naive_prediction(testset)
    y_true = [testset[i][2] for i in range(len(testset))]
    k_rmse.append(mean_squared_error(y_true,y_pred))
avg_rmse= np.mean(k_rmse)
print('The average RMSE using naive collaborative filter is %0.4f'%avg_rmse)

#Question 31
k_rmse = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    testset = popular_trim(testset)
    y_pred = naive_prediction(testset)
    y_true = [testset[i][2] for i in range(len(testset))]
    k_rmse.append(mean_squared_error(y_true,y_pred))
avg_rmse= np.mean(k_rmse)
print('The average RMSE for popular movie trimmed set is %0.4f'%avg_rmse)

#Question 32
k_rmse = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    testset = unpopular_trim(testset)
    y_pred = naive_prediction(testset)
    y_true = [testset[i][2] for i in range(len(testset))]
    k_rmse.append(mean_squared_error(y_true,y_pred))
avg_rmse= np.mean(k_rmse)
print('The average RMSE for unpopular movie trimmed set is %0.4f'%avg_rmse)

#Question 33
k_rmse = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    testset = high_var_trim(testset)
    y_pred = naive_prediction(testset)
    y_true = [testset[i][2] for i in range(len(testset))]
    k_rmse.append(mean_squared_error(y_true,y_pred))
avg_rmse= np.mean(k_rmse)
print('The average RMSE for high variance movie trimmed set is %0.4f'%avg_rmse)

#Question 24
k_range = range(2,50,2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = SVD(n_factors=k)
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))
        
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.xlabel('Number of latent factors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' %(np.min(avg_rmse),np.argmin(avg_rmse)))

#Question 25
k_range = range(2,51,2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = SVD(n_factors=k)
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(popular_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))
        
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.xlabel('Number of latent factors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' %(np.min(avg_rmse),np.argmin(avg_rmse)))  


def get_top_t(predictions, t=10):
    # First map the predictions to each user.
    top_t = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_t[uid].append((iid, est, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_t.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_t[uid] = user_ratings[:t]

    return top_t  
     
train_set, test_set = train_test_split(data, test_size=0.1, random_state=0)
algo = KNNWithMeans(k=20, sim_options = {'name':'pearson'}) 
algo.fit(train_set)
predictions = algo.test(test_set)
top_recos = get_top_t(predictions)

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precisions, recalls

#Quesion 36   
kf = KFold(n_splits=10)
algo = KNNWithMeans(k=20, sim_options = {'name':'pearson'}) 
threshold = 3

avg_prec, avg_rec = [], []

for t in range(1, 26):
    t_prec, t_rec = [], []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=t, threshold=threshold)
        t_prec.append((sum(prec for prec in precisions.values()) / len(precisions)))
        t_rec.append(sum(rec for rec in recalls.values()) / len(recalls))
    avg_prec.append(np.mean(t_prec))
    avg_rec.append(np.mean(t_rec))    

t_range = range(1,26)

plt.plot(t_range, avg_prec)
plt.xlabel('Item set size t', fontsize=15)
plt.ylabel('Average Precision', fontsize=15)
plt.title("Precision vs t for KNN")
plt.show()

plt.plot(t_range, avg_rec)
plt.xlabel('Item set size t', fontsize=15)
plt.ylabel('Average Recall', fontsize=15)
plt.title("Recall vs t for KNN")
plt.show()

plt.plot(avg_rec, avg_prec)
plt.xlabel('Average Recall', fontsize=15)
plt.ylabel('Average Precision', fontsize=15)
plt.title("Precision vs Recall for KNN")
plt.show()

#Quesion 37 
kf = KFold(n_splits=10)
algo = NMF(n_factors=20)
threshold = 3

avg_prec, avg_rec = [], []

for t in range(1, 26):
    t_prec, t_rec = [], []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=t, threshold=threshold)
        t_prec.append((sum(prec for prec in precisions.values()) / len(precisions)))
        t_rec.append(sum(rec for rec in recalls.values()) / len(recalls))
    avg_prec.append(np.mean(t_prec))
    avg_rec.append(np.mean(t_rec))    

t_range = range(1,26)

plt.plot(t_range, avg_prec)
plt.xlabel('Item set size t', fontsize=15)
plt.ylabel('Average Precision', fontsize=15)
plt.title("Precision vs t for KNN")
plt.show()

plt.plot(t_range, avg_rec)
plt.xlabel('Item set size t', fontsize=15)
plt.ylabel('Average Recall', fontsize=15)
plt.title("Recall vs t for KNN")
plt.show()

plt.plot(avg_rec, avg_prec)
plt.xlabel('Average Recall', fontsize=15)
plt.ylabel('Average Precision', fontsize=15)
plt.title("Precision vs Recall for KNN")
plt.show()

#Quesion 38

kf = KFold(n_splits=10)
algo = SVD(n_factors=20, init_mean = 3)
threshold = 3

avg_prec, avg_rec = [], []

for t in range(1, 26):
    t_prec, t_rec = [], []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=t, threshold=threshold)
        t_prec.append((sum(prec for prec in precisions.values()) / len(precisions)))
        t_rec.append(sum(rec for rec in recalls.values()) / len(recalls))
    avg_prec.append(np.mean(t_prec))
    avg_rec.append(np.mean(t_rec))    
    
t_range = range(1,26)

plt.plot(t_range, avg_prec)
plt.xlabel('Item set size t', fontsize=15)
plt.ylabel('Average Precision', fontsize=15)
plt.title("Precision vs t for SVD")
plt.show()

plt.plot(t_range, avg_rec)
plt.xlabel('Item set size t', fontsize=15)
plt.ylabel('Average Recall', fontsize=15)
plt.title("Recall vs t for SVD")
plt.show()

plt.plot(avg_rec, avg_prec)
plt.xlabel('Average Recall', fontsize=15)
plt.ylabel('Average Precision', fontsize=15)
plt.title("Precision vs Recall for SVD")
plt.show()

#Question 39
kf = KFold(n_splits=5)
algos = [KNNWithMeans(k=20, sim_options = {'name':'pearson'}), NMF(n_factors = 20),
                      SVD(n_factors=20, init_mean = 2.5)]
threshold = 3
algo_prec, algo_rec = [], [] 
for algo in algos:
    avg_prec, avg_rec = [], []
    for t in range(1,26):
        t_prec, t_rec = [], []
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = precision_recall_at_k(predictions, k=t, threshold=threshold)
            t_prec.append((sum(prec for prec in precisions.values()) / len(precisions)))
            t_rec.append(sum(rec for rec in recalls.values()) / len(recalls))
        avg_prec.append(np.mean(t_prec))
        avg_rec.append(np.mean(t_rec)) 
    algo_prec.append(avg_prec)
    algo_rec.append(avg_rec)
    
plt.plot(algo_rec[0], algo_prec[0], lw = 2, label = 'KNN')
plt.plot(algo_rec[1], algo_prec[1], lw=2, label = 'NNMF')
plt.plot(algo_rec[2], algo_prec[2], lw = 2, label = 'MF with bias')
plt.xlabel('Average Recall', fontsize=15)
plt.ylabel('Average Precision', fontsize=15)
plt.title("Precision vs Recall for different algorithms")
plt.legend()
plt.show()
    
#Question 34
train_set, test_set = train_test_split(data, test_size=0.1, random_state=0)
threshold = 3 
algos = [KNNWithMeans(k=20, sim_options = {'name':'pearson'}), NMF(n_factors = 20),
                      SVD(n_factors=20, init_mean = 2.5)]
algo_fpr,algo_tpr,algo_auc = [],[],[]
for algo in algos:
    print('Algorithm is ', algo)
    algo.fit(train_set)
    predictions = algo.test(test_set)
    pred_est = np.array([i.est for i in predictions])
    actual_ratings = np.array([i.r_ui for i in predictions])
    y_score = pred_est
    y_true = actual_ratings>=threshold
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    algo_fpr.append(fpr)
    algo_tpr.append(tpr)
    algo_auc.append(roc_auc)
    
plt.plot(algo_fpr[0], algo_tpr[0], lw=2, label='KNN, AUC = %0.4f' %algo_auc[0])
plt.plot(algo_fpr[1], algo_tpr[1], lw=2, label='NNMF, AUC = %0.4f' %algo_auc[1])
plt.plot(algo_fpr[2], algo_tpr[2], lw=2, label='MF with bias, AUC = %0.4f' %algo_auc[2])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate')
plt.legend(loc ='lower right')
plt.title('ROC curves for different algorithms')
plt.show()
