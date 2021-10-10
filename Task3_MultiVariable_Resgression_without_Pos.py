# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:16:36 2021

@author: favou
"""

import pandas as pd 

from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



from Testing_heteroskedasticity import Breushpagan_test




#Data
# df = pd.read_csv("Nba_Regression_data.csv")
df = pd.read_csv("Nba_Regression_data_new1.csv")



#######################  Modeling only numeric variable  #####################################################

numeric_df = df._get_numeric_data()
X = numeric_df.drop(['TOV'], axis =1)
X_sm = sm.add_constant(X)
y = numeric_df['TOV']

X_train, X_test, y_train, y_test = train_test_split(X_sm, y, test_size = 0.2, random_state = 0)

result = sm.OLS(y_train, X_train).fit() #statsmodel: linear model
lm = linear_model.LinearRegression().fit(X_train, y_train)
result.summary()

y_pred_test = result.predict(X_test)

plt.figure()
sns.scatterplot(y_test, y_pred_test, marker = '+', ci = False, alpha = 0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = 'orange', label = 'y=x line')
plt.ylabel('Predicted TOV') 
plt.xlabel('Actual TOV')
plt.legend()

coefficients = result.params[1:len(result.params)]
intercept = result.params[0]

print("Regression Model with only numeric data: ")
print(f"The coefficients(2PA, FTA, AST) are {[round(coefficient, 3) for coefficient in coefficients]}")
print(f"The intercept is {intercept:.3f}")

print("")
print("Evaluation")
print("1) Testing for Heteroskedasticity")
Breushpagan_test(result)

print("2) Errors and R2")
#Evaluating model
    #1) MAE (A measure of average error between best fit curve and data that doesn't penalise highly innacurate predicts )

print(f"The mean absolute error is: {mean_absolute_error(y_test, y_pred_test):.3f}")
    #2) RMSE (A measure of average error between best fit curve and data that penalises highly innacurate predicts (as error is squared instead of just modulus)) )
rmse =  mean_squared_error(y_test, y_pred_test)**0.5  

print(f"The root mean squared error is : {rmse:.3f}")

        #2A) Determining the severity of the error 
from scipy import stats
percentile = stats.percentileofscore(numeric_df.TOV, rmse) 
        # 'percentile' gives the percentage of TOV observations that the 'rmse' 
        #is greater than or equal to. The inverse of this would be finding the 
        #quantile (i.e. the TOV value that a given percentage of TOV observations 
        #fall at or below (e.g. quantile = df.quantile(percentile/100) = rmse
print(f'The rmse is greater than or equal to the {percentile:.3f} percentile of TOV observations ({percentile*0.01*len(numeric_df)} out of {len(numeric_df)} observations)')
  
    #3) R2 (measures how well the best-fit curve fits data)
print(f" The R_squared is: {r2_score(y_test, y_pred_test):.3f}")


################## Findings and comments 

#The coefficients(2PA, FTA, AST) are [0.048, 0.123, 0.166]
#The intercept is 0.706

#For the predicted v test data the following metrics were found: 
    #1)The mean absolute error is: 0.405
    #2)The root mean squared error is : 0.529 (belongs to the 0.27percentile of TOV observations)
     #i.e. the RMSE is greater than or equal to 0.27% of the TOV data  (from numeric_df['TOV'].median() or numeric_df['TOV'].quantile(0.5) ))
    #3)The R_squared is: 0.464 (compared to the R_squared for the train data set = 0.465 (from summary table))

#All chosen variables are significant (P>|t|< 0.05)

#The residuals are not normally distributed but a lot of data is used of this is unlikely:
    #P(omnibus) = 0 (for normal dist =1)
    #P(JB) = 0 (for normal dist >0.05)

#Residuals is not autocorrelated Durbin-Watson = 1.988

#Residual has a slight positive skew (=0.836)
#Residual is also displays some leptokurtic(kurtosis =5.957 taller and thinner than normal dist curve)

#Cond. No. = 34.9 (which is low  so there is no multicolinearity )



#############################Making Predictions ###############################
Two_PA = 9.85987 #number of two-point shots taken 
FTA = 4.70064 #number of free throws attempted
Assists = 6.99363 #number of asists 

Turnover = intercept + coefficients['AST']*Assists + coefficients['2PA']*Two_PA + coefficients['FTA']*FTA


print(f"A player that takes {Two_PA} two-point shots and has  {Assists} assists and {FTA} freethrow attempts may commit {Turnover:.1f} turnovers")

#############################################################################################################


##### Residual 
# Checking residuals (training dataset)
y_pred = result.predict(X_train)

residual_df = y_train-y_pred
plt.figure()
plt.scatter(y_pred,residual_df, marker = '+' )
plt.plot([0,4], [0,0], c = 'black')
#plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.xlim(0.7,4)
plt.show()


# Checking Normality of errors: histogram plot
residual_df.mean()
plt.figure()
sns.distplot(residual_df)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
#####
##############

######

# Checking Normality of errors: Q-Q plot

from scipy.special import ndtri

df_QQ = pd.DataFrame(residual_df.sample(frac = 0.1).values, columns = ['Residual'])
df_QQ = df_QQ.sort_values(by=['Residual'], ascending = True).reset_index()[['Residual']] 
df_QQ['Count'] = df_QQ.index +1
df_QQ['Percentile_Area']= df_QQ.Count/df_QQ.shape[0]
df_QQ['Z_theory'] = ndtri(df_QQ.Percentile_Area) 
df_QQ['Z_actual'] = (df_QQ.Residual - df_QQ.Residual.mean())/df_QQ.Residual.std(ddof =0)

plt.figure()
plt.scatter(df_QQ.Z_theory, df_QQ.Z_actual, marker = '+', c = 'black', s = 10)
plt.xlabel('Thoeretic quantile')
plt.ylabel('Sample(10%) quantile')
plt.plot([-5, 7], [-5, 7])

plt.figure()
plt.scatter(df_QQ.Z_theory, df_QQ.Z_theory)
plt.plot([-5, 7], [-5, 7])


##############################Saving Model######################################
# import pickle

#     #stats model 
# with open('Linear_Regrssion_Model_Numeric_columns_SM.sav','wb') as f:
#       pickle.dump(result,f)
   
#     #sklearn model 
# with open('Linear_Regrssion_Model_Numeric_columns_sk.sav','wb') as f:
#       pickle.dump(lm,f)
################################################################################



      
      
      