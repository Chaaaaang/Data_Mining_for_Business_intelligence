# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:12:10 2018

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def all_scatter(df, group=None, size=3):
    """
    seaborn을 사용하여 모든 변수들의 상관관계 그래프와 각 변수의 히스토그램을 구합니다.
    그룹 값을 넣으면 그룹별로 산점도를 구합니다.
    (seaborn의 pairplot 사용)
    """
    return sns.pairplot(df, hue=group, size=size)
    
def groupby_bar(df, g_column, target_variable, method, width=10, height=10):
    """
    특정 변수 그룹을 기준으로 변수값들의 평균이나 % 정도를 구합니다.
    """
    if method.lower() == 'a':
        return df.groupby(g_column).mean()[target_variable].plot.bar()
    elif method.lower() == 'p':
        return ((df.groupby(g_column).count()[target_variable])/(len(df[target_variable]))).plot.bar()
    else :
        print('잘못된 값이 입력되었습니다.')
    
def all_boxplot(df, width=10, height=10):
    """
    모든 변수의 boxplot을 구합니다.
    """
    plt.figure(figsize=(width,height))
    return df.boxplot(return_type='axes')

def df_hist(df, variable, width=10, height=10):
    """
    변수의 히스토그램을 그립니다.
    """
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    return df[variable].hist(figsize=(width, height))

def groupby_boxplot(df, g_column, target_variable, width=10, height=10):
    """
    그룹별로 타겟 변수의 값들을 구합니다.
    """
    return df.boxplot(column=target_variable, by=g_column, return_type='axes',
                      figsize=(width, height))
        
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
    
def corr_general_heatmap(df, width=10, height=10):
    """
    seaborn을 이용한 상관관계를 나타내는 heatmap을 만듭니다.
    """
    corr = df.corr()
    plt.figure(figsize = (width, height))
    sns.heatmap(corr, annot=True, cmap="YlGnBu")
    
def missing_check_heatmap(df, width=10, height=10):
    """
    결측값이 있는 부분을 하얀색으로 표시하는 heatmap을 만듭니다.
    df.info()와 같은 역할을 합니다.
    """
    mask = df.isnull()
    plt.figure(figsize = (width, height))
    sns.heatmap(df, mask=mask, cmap='viridis')
        
def multi_panel_plot(df, g_column, x, y, width=10, height=10):
    """
    패널이 분리된 차트를 만듭니다. 
    즉, 특정 그룹 변수를 기준으로 x값은 카테고리형, y값은 x 카테고리 별 y의 평균값입니다.
    """
    plt.figure(figsize = (width, height))
    rad_group = df.groupby([g_column, x]).mean().unstack(g_column)
    ax = rad_group[y].plot.bar(logx=True, logy=True)
    ax.set_ylabel('avg('+y+')')
    
def scale_scatter(df, x, y, width=15, height=10):
    """
    위의 창에 조정 전 값과 아래 창에 로그로 축을 조절한 산포도 그래프를 만듭니다.  
    """
    fig, axes = plt.subplots(2, figsize=(x, y))
    axes[0].scatter(df[x], df[y])
    axes[0].set_xlabel(x)
    axes[1].scatter(df[x], df[y])
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    plt.show()
    
def scale_boxplot(df, x, y, width=10, height=10):
    """
    위의 창에 조정 전 값과 아래 창에 로그로 축을 조절한 박스 그래프를 만듭니다.
    """
    fig, axes = plt.subplots(2, figsize=(width,height))
    plt.subplot(221)
    sns.boxplot(x=x, y=y, data=df)
    plt.subplot(223)
    sns.boxplot(x=x, y=y, data=df)
    plt.yscale('log')
    plt.show()
    
def parallel_coor(df, g_column, width=20,height=10):
    """
    그룹을 기준으로 나누는 평행 좌표를 만듭니다.
    """
    g_set = set(df[g_column])
    a = df.columns.values.tolist()
    mat_num = 0
    if np.sqrt(mat_num) == int(np.sqrt(mat_num)):
        mat_num = int(np.sqrt(len(g_set)))
    else:
        mat_num = int(np.sqrt(len(g_set)))+1
    mat_num_start = 1
    plt.figure(figsize=(width,height))
    plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
    for k in g_set:
        plt.subplot(mat_num, mat_num, mat_num_start)
        plt.plot([[i]*len(df[df[g_column]==k]) for i in range(len(df.columns.values))], 
                  [(df[df[g_column]==k].iloc[:,i].values-min(df.iloc[:,i].values))
                  /(max(df.iloc[:,i].values)-min(df.iloc[:,i].values))*100 
                  for i in range(len (df.columns))])
        plt.xticks(range(len(a)),a)
        plt.ylabel("(%)")
        plt.title(str(g_column)+" = "+str(k))
        mat_num_start = mat_num_start+1
        
path = r'C:\Users\User\Desktop\Boston/'
#path = r'C:\Users\Lee\Desktop\Boston/'
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv(path+'boston_train.csv')
amtrak_data = pd.read_csv(path+'amtrak.csv', index_col='Date')
amtrak_data.columns = ['Ridership(raw)', 'Ridership']
# 만약 medv>30 : cat_medv가 1인 데이터 추가(반대는 0이 됨)
temp = data.medv>30
temp[temp == False] = 0
temp[temp == True] = 1
data['cat_medv'] = temp

all_boxplot(data, width=30, height=20)
df_hist(data,'medv')
missing_check_heatmap(data)
