# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:21:50 2018

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def corr_white_heatmap(df, width=10, height=10):
    """
    하부삼각행렬로 구성된 상관관계를 나타내는 heatmap을 만듭니다.
    참고사이트 : https://seaborn.pydata.org/generated/seaborn.heatmap.html
    """
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize = (width, height))
    sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True)


path = r'C:\Users\User\Desktop\Boston/'
#path = r'C:\Users\Lee\Desktop\Boston/'
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv(path+'boston_train.csv')

# 만약 medv>30 : cat_medv가 1인 데이터 추가(반대는 0이 됨)
temp = data.medv > 30
temp[temp == False] = 0
temp[temp == True] = 1
data['cat_medv'] = temp

# 데이터 요약(통계량)
data.info()
data.describe()

# heatmap 그리기
corr_white_heatmap(data)

# 피봇테이블 : chas별 medv 개수
pd.pivot_table(data, values='medv', index='chas', aggfunc=len)
# 피봇테이블 : rm 구간별 chas에 따른 평균 medv 값
data.pivot_table(values='medv', index=pd.cut(data.rm,[3,4,5,6,7,8,9]), 
                     columns='chas', aggfunc=np.mean)

# list(set(data.zn)) = data.zn.unique()) 랑 같은 기능을 합니다.
"""
data.pivot_table(values='cat_medv', index='zn').plot.bar(color='r')
data.pivot_table(values='cat_medv', index='zn', aggfunc=lambda x: 1-np.mean(x)).plot.bar(
        color='b',bottom=list(set(data.zn)))
"""

# 데이터 변환 : zn에 의한 cat_medv의 분포
pd.concat([data.pivot_table(values='cat_medv', index='zn'),
          data.pivot_table(values='cat_medv', index='zn', aggfunc=lambda x: 1-np.mean(x))], axis=1).plot(kind='bar',stacked=True)

# 위의 2줄 코드랑 같은 기능을 하는 코드입니다. (agg 함수 : 2개 이상 함수 쓸 때 사용)
ax = data.groupby('zn')['cat_medv'].agg([np.mean,lambda x:1-np.mean(x)]).plot(kind='bar', stacked=True)
ax.legend(['0','1'], loc='upper right')


