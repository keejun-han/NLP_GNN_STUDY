# Created on Feb 2020
# Author: 임일

import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)            # timestamp 제거

# train test 분리
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=1)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

# RMSE 계산을 위한 함수
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

##### (1)

# Dummy recommender 0
def recommender0(recomm_list):
    recommendations = []
    for pair in recomm_list:
        recommendations.append(random.random() * 4 + 1)
    return np.array(recommendations)

# Dummy recommender 1
def recommender1(recomm_list):
    recommendations = []
    for pair in recomm_list:
        recommendations.append(random.random() * 4 + 1)
    return np.array(recommendations)

# Hybrid 결과 얻기
weight = [0.8, 0.2]
recomm_list = np.array(ratings_test)
predictions0 = recommender0(recomm_list)
predictions1 = recommender1(recomm_list)
predictions = predictions0 * weight[0] + predictions1 * weight[1]
RMSE2(recomm_list[:, 2], predictions)

