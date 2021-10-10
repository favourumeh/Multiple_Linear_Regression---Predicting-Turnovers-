# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 11:17:12 2021

@author: favou
"""

import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_excel('1990_2021_NBA_ppg_stats.xlsx')

df.shape

df.head()
df.tail()

df.columns

df1 =df[['Pos', 'Age', 'MP', '2PA', '2P_per', '3PA', '3P_per', 'FTA','FT_per','TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']] 
A = df1.describe()


df.info()
df.dtypes

df.isnull().any()
df.isnull().sum() #number of column that has null values 
df.isnull().sum()/(df.shape[0]*0.01) #percentatge of column that has null values 

df.Pos.value_counts() # number of each type of player featured

df.Pos.notnull().sum() # sum of fields that are not null in column 'Pos'


#Simple Plotting 
df.Pos.value_counts().plot(kind = 'bar', color = {'blue','purple','orange','green', 'red' })
plt.xlabel('Player Positions')
plt.ylabel('No. of occurences')


df.Age.hist(bins =27)

