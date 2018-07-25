# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:55:02 2018

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from numpy import linalg as LA

path = r'C:\Users\User\Desktop\Boston/'
data = pd.read_csv(path+'cereal.txt', sep='\t')
print(data.head())
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize = (10, 10))
sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True)

# calories, rating의 공분산 행렬
np.cov(data.calories, data.rating)

# 변수의 평균을 0으로 만드는 작업
x1 = data.calories-np.mean(data.calories)
y1 = data.rating-np.mean(data.rating)

# 고유값 구하는 코드
c_value = np.cov(x1,y1)
e_value, e_vector = np.linalg.eig(c_value)
# 고유값
print(e_value)
# 고유벡터
print(e_vector)

# 고유값의 그래프 그리기
plt.scatter(x1, y1)
plt.quiver(0,0,50*e_vector[1,0],50*e_vector[1,1],angles='xy', scale_units='xy', scale=1)
plt.quiver(0,0,20*e_vector[0,0],20*e_vector[0,1],angles='xy', scale_units='xy', scale=1)
plt.show()


X = pd.DataFrame({'calories':data.calories,'rating':data.rating})
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X.iloc[:,0].reshape(-1,1))
plt.scatter(X_pca,np.zeros_like(X_pca))
print(X_pca)

# cereal의 수치형 변수를 2차원으로 표현하기
# fit : 적합시킴 
model = PCA(2).fit(data.iloc[:,4:])
# transform : 데이터를 2차원으로 변형시킴
X_pcl = model.transform(data.iloc[:,4:])
# 그래프 그리기
plt.scatter(X_pcl[:,0], X_pcl[:,1], alpha=0.9)
plt.axis('equal');
# 설명된 분산 비율
model.explained_variance_ratio_
# 각 주성분의 설명된 분산값을 원래 있던 변수가 차지하는 가중치를 구함.
model.components_


sum(data.iloc[:,6]*(data.iloc[:,6]-np.mean(data.iloc[:,6])))

"""
(참고) inverse_transform : 제일 큰 분산만 남기고 나머지 오차를 제거한 값들을 기준으로 차원 복구
(예시) X_new = model.inverse_transform(model.transform(data.iloc[:,4:]))
"""