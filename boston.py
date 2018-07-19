# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:12:10 2018

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path = r'C:\Users\User\Desktop\Boston/'
#path = r'C:\Users\Lee\Desktop\Boston/'
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv(path+'boston_train.csv')

# 만약 medv>30 : cat_medv가 1인 데이터 추가(반대는 0이 됨)
temp = data.medv>30
temp[temp == False] = 0
temp[temp == True] = 1
data['cat_medv'] = temp

# 참고 홈페이지
# https://datascienceschool.net/view-notebook/d0b1637803754bb083b5722c9f2209d0/
# 2*2에서 1번째를 뜻함
# = plt.subplot(2,2,1)
plt.figure(figsize=(15,15))
plt.subplot(221)
plt.scatter(data.lstat,data.medv)
plt.ylabel('MEDV')
plt.xlabel('LSTAT')

plt.subplot(222)
average_medv0 = np.mean(data.medv[data.chas==0])
average_medv1 = np.mean(data.medv[data.chas==1])
average_medv_list = [average_medv0,average_medv1]
plt.ylabel('Average MEDV')
plt.xlabel('CHAS')
plt.bar(list(set(data.chas)),average_medv_list)
# x를 0,1로만 범위 정함
plt.xticks([0, 1])

plt.subplot(223)
per_cat_chas0 = len(data[data.chas==0])/len(data)
per_cat_chas1 = len(data[data.chas==1])/len(data)
per_cat_chas_list = [per_cat_chas0, per_cat_chas1]
plt.bar(list(set(data.chas)),per_cat_chas_list)
plt.xticks([0, 1])
plt.ylabel('% of CAT.MEDV')
plt.xlabel('CHAS')


medv0 = data.medv[data.chas==0]
medv1 = data.medv[data.chas==1]

fig, axes = plt.subplots(3, 2, figsize=(15,15))
axes[0, 0].hist(data.medv)
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_xlabel("MEDV")
axes[0, 1].boxplot([medv0,medv1])
axes[0, 1].set_ylim(min(data.medv)-np.std(data.medv),
    max(data.medv)+np.std(data.medv))
axes[1, 0].boxplot([data.ptratio[data.cat_medv==0],data.ptratio[data.cat_medv==1]])
axes[1, 0].set_ylim(min(data.ptratio)-np.std(data.ptratio),
    max(data.ptratio)+np.std(data.ptratio))
axes[1, 1].boxplot([data.nox[data.cat_medv==0],data.nox[data.cat_medv==1]])
axes[1, 1].set_ylim(min(data.nox)-np.std(data.nox),
    max(data.nox)+np.std(data.nox))
axes[2, 0].boxplot([data.indus[data.cat_medv==0],data.indus[data.cat_medv==1]])
axes[2, 0].set_ylim(min(data.indus)-np.std(data.indus),
    max(data.indus)+np.std(data.indus))
axes[2, 1].boxplot([data.lstat[data.cat_medv==0],data.lstat[data.cat_medv==1]])
axes[2, 1].set_ylim(min(data.lstat)-np.std(data.lstat),
    max(data.lstat)+np.std(data.lstat))
plt.show()

# heatmap1 : 하부삼각행렬로만 보여줌
# 참고사이트 : https://seaborn.pydata.org/generated/seaborn.heatmap.html
corr = data.iloc[:,1:].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize = (10,10))
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True)

# heatmap2 : 상,하부 삼각행렬 전부 포함
plt.figure(figsize = (10,10))
sns.heatmap(corr, annot=True, cmap="YlGnBu")

# heatmap3 : 결측값 표시
mask = data.iloc[:,1:].isnull()
sns.heatmap(data.iloc[:,1:], mask=mask, cmap='viridis')

"""
# heatmap4 : 결측값 예시
data2 = data.copy()
data2.chas[data2.chas==0] = np.NaN
mask = data2.iloc[:,1:].isnull()
sns.heatmap(data2.iloc[:,1:], mask=mask, cmap="YlGnBu")
"""

# scatter with color point
plt.figure(figsize = (10,10))
plt.scatter(data.lstat[data.cat_medv==0],
            data.nox[data.cat_medv==0], 
            c=['r' for i in range(len(data[data.cat_medv==0]))],
            label='cat_medv=0')
plt.scatter(data.lstat[data.cat_medv==1],
            data.nox[data.cat_medv==1], 
            c=['b' for i in range(len(data[data.cat_medv==1]))],
            label='cat_medv=1')
plt.xlabel('LSTAT', fontsize=18)
plt.ylabel('NOX', fontsize=18)
plt.legend(loc='upper right')
plt.grid()
plt.show()

# multi panel
plt.figure(figsize = (15,10))
rad_group = data.groupby(['chas','rad']).mean().unstack('chas')
rad_group['medv'].plot.bar()
plt.show()

# scatter matrix
sns.pairplot(data, diag_kind='kde')

# 2x2 같이 열 단위로 나눌 때는 최소 2x2가 되어야 나눌 수 있음
# axis scale change : scatter
fig, axes = plt.subplots(2, figsize=(15,10))
axes[0].scatter(data.crim, data.medv)
axes[1].scatter(data.crim, data.medv)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
plt.show()

# axis scale change : boxplot
# seaborn으로 만들면 밑의 숫자도 변형 할 필요 없음
fig, axes = plt.subplots(2, figsize=(15,10))
plt.subplot(221)
sns.boxplot(x='cat_medv', y='crim',data=data)
plt.subplot(223)
sns.boxplot(x='cat_medv', y='crim',data=data)
plt.yscale('log')
plt.show()

amtrak_data = pd.read_csv(path+'amtrak.csv', index_col='Date')
amtrak_data.columns = ['Ridership(raw)', 'Ridership']
date_list = []
s_year = 1991
s_month = 1
s_day = '01'
for i in range(len(amtrak_data)):
    if i == 0:
        date_list.append(str(s_year)+'-0'+str(s_month)+'-'+s_day)
        continue
    if i%12==0 :
        s_month=1
        s_year=s_year+1
    if s_month>=10:
        date_list.append(str(s_year)+'-'+str(s_month)+'-'+str(s_day))
    else:
        date_list.append(str(s_year)+'-0'+str(s_month)+'-'+str(s_day))
    s_month=s_month+1
date_list = pd.to_datetime(pd.Series(date_list))
amtrak_data.index = date_list

# 시계열 데이터 만들기
plt.figure(figsize=(15,10))
# 쉼표 제거 및 타입 변경 : string의 경우 str을 붙이고 replace 할 것
amtrak_data.Ridership = amtrak_data.Ridership.str.replace(',','').astype(float)

# 1991/1/1 ~ 2004/3/1
ax = amtrak_data.Ridership.plot()
ax.set_xlabel('Date')
ax.set_ylabel('Ridership')
ax.grid()

# 평행 좌표계
# [1,1]과 [2,6]과 [5,1], [2,2]와 [5,7]과 [6,2] 등등...
# plt.plot([[1,2,3],[2,5,1],[5,6,8]],[[1,2,3],[6,7,2],[1,2,5]])
a = data.columns.values[1:].tolist()
plt.figure(figsize = (25,10))
plt.subplot(221)
plt.plot([[i]*len(data[data.cat_medv==0]) for i in range(len(data.columns.values[1:]))], 
          [(data[data.cat_medv==0].iloc[:,i].values-min(data.iloc[:,i].values))
          /(max(data.iloc[:,i].values)-min(data.iloc[:,i].values))*100 
           for i in range(len(data.columns[1:]))])
plt.xticks(range(15),a)

plt.figure(figsize = (25,10))
plt.subplot(222)
plt.plot([[i]*len(data[data.cat_medv==1]) for i in range(len(data.columns[1:]))], 
          [(data[data.cat_medv==1].iloc[:,i].values-min(data.iloc[:,i].values))
          /(max(data.iloc[:,i].values)-min(data.iloc[:,i].values))*100 
           for i in range(len(data.columns[1:]))])
plt.xticks(range(15),a)
plt.show()