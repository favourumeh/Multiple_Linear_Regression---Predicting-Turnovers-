# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 18:33:35 2021

@author: favou
"""
from sklearn.model_selection import cross_val_score
import pandas as pd
import pickle 

# Model file names:
    # Model1_sklearn.sav #sklearn model
    # Model1_statsmodels.sav #statsmodels model
    
    # Model2_sklearn.sav
    # Model2_statsmodels.sav



df = pd.read_csv("Nba_Regression_data_new1.csv")

# Model 1
df1 = df._get_numeric_data()


# Model 2
df1 = pd.get_dummies(df) 


# Variables 
X = df1.drop(['TOV'], axis =1)
X['const'] = 1
y = df1['TOV']



#loading model
with open('Model2_sklearn.sav','rb') as f:
      lm = pickle.load(f)


#Cross Validation: calculate RMSE
RMSE = cross_val_score(lm, X, y, cv = 5, scoring = 'neg_root_mean_squared_error')
RMSE_mean = RMSE.mean()

