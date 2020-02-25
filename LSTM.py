#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 21:46:25 2019

@author: 颜一凡
"""

"""
    要顺利运行此模型一定要用我的train.csv 该文件含有test.csv的数据 
    只是为了读数据方便!!!直接一次索引即可
    没有使用测试集训练!!!
    cross-validation的过程完全在训练集中进行
    
    此预测没有用测试集的任何数据训练
    将测试集的数据放在训练集组成 train.csv 只是为了读文件方便 
    读完文件后测试集数据仅用于最终计算smape
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

train = pd.read_csv('./train.csv').fillna(0)
page = train['Page']

def smape(pred,test):
    s=0
    count=0
    for i in range(len(test)):
        if test[i] == 0:
            continue
        else:
            a=pred[i]
            b=test[i]
            s+= math.fabs(a-b)/((a+b)/2)
            count+=1
    print(count,end='\t')
    return s/count
            


train = train.drop('Page',axis = 1)

ans=np.zeros((100,60))
total=0
num=2


for k in range(100):
    row = train.iloc[k,:].values
    
    cycle=60
    
    """
    获得训练集 和 测试集
    测试集在后面仅用来算smape，未参与训练
    """
    X_train=row[0:609-cycle]
    X_test=row[609-cycle:609]
    y_train=row[1:610-cycle]
    y_test=row[610-cycle:610]
    
    sc = MinMaxScaler()
    X_train = np.reshape(X_train,(-1,1))
    y_train = np.reshape(y_train,(-1,1))
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)
    
    l=len(X_train)
    X_train = np.reshape(X_train, (l,1,1))
    y_pred_best=0
    
            
    regressor = Sequential()
    
    # 给LSTM网络加入隐藏层
    regressor.add(LSTM(units = 8, activation = 'relu', input_shape = (None, 1)))
    
    
    # 添加输出层
    regressor.add(Dense(units = 1))
    
    # 用adam优化算法 和 误差平方和 作为损失函数
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    
    """
    用训练集来训练网络 完全无关测试集
    """
    regressor.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose = 0)
    
    inputs = X_test
    inputs = np.reshape(inputs,(-1,1))
    inputs = sc.transform(inputs)
    ll=len(inputs)
    inputs = np.reshape(inputs, (ll, 1, 1))
    y_pred = regressor.predict(inputs)
    y_pred = sc.inverse_transform(y_pred)
    
    y_test=y_test[cycle-60:]
    y_pred=y_pred[cycle-60+1:]
    y_pred=np.vstack((y_pred,[np.mean(y_pred)]))
    y_pred=np.round(y_pred,2)
    
    for index in range(60):
        if y_pred[index]<0:
            y_pred[index]=-y_pred[index]

    plt.figure(figsize=(20,10))
    plt.cla()
    plt.plot(y_test, color = 'red', label = 'Real Web View')
    plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
    plt.title('Web View Forecasting')
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    plt.show()
    

    mape=smape(y_pred,y_test)
    total+=mape
    print(k,mape)
    for index in range(60):
        ans[k][index]=y_pred[index]
    
ans_arr=pd.DataFrame(ans)
ans_arr.to_excel('myAns4.xlsx')
print(total/100)

