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
df = pd.read_csv("Nba_Regression_data_new1.csv")


######################  Adding Categoric data ##################################

#converting categoric variable to numeric 
df1 = pd.get_dummies(df)


X_c = df1.drop(['TOV'], axis =1)
X_c = sm.add_constant(X_c)
y_c = df1.TOV


X_train1, X_test1, y_train1, y_test1 = train_test_split(X_c, y_c, test_size = 0.2, random_state = 0)

result1 = sm.OLS(y_train1, X_train1).fit()
result1.summary()

lm = linear_model.LinearRegression().fit(X_train1, y_train1)

coefficients1 = round(result1.params[1:len(result1.params)], 3)
intercept1 = round(result1.params[0], 3)



print("Regression Model with Categoric data: ")
print(coefficients1)
print('intercept', intercept1)
# print(f"The coefficients(2PA, AST, Pos_C, Pos_PF, Pos_PG, Pos_SF, Pos_SG ) are {[round(coefficient, 3) for coefficient in coefficients1]}")
# print(f"The intercept is {intercept1:.3f}")

y_pred_test1 = result1.predict(X_test1)

plt.figure()
sns.scatterplot(y_test1, y_pred_test1, marker = '+', ci = False, alpha = 0.5)
plt.plot([min(y_test1), max(y_test1)], [min(y_test1), max(y_test1)], color = 'orange', label = 'y=x line')
plt.ylabel('Predicted TOV') 
plt.xlabel('Actual TOV')
plt.legend()

print("")
print("Evaluation")
print("1) Testing for Heteroskedasticity")
Breushpagan_test(result1)

print("2) Errors and R2")
#Evaluating model
    #1) MAE (A measure of average error between best fit curve and data that doesn't penalise highly innacurate predicts )

print(f"The mean absolute error is: {mean_absolute_error(y_test1, y_pred_test1):.3f}")
    #2) RMSE (A measure of average error between best fit curve and data that penalises highly innacurate predicts (as error is squared instead of just modulus)) )
rmse =  mean_squared_error(y_test1, y_pred_test1)**0.5  

print(f"The root mean squared error is : {rmse:.3f}")

        #2A) Determining the severity of the error 
from scipy import stats
percentile = stats.percentileofscore(df1.TOV, rmse) 
        # 'percentile' gives the percentage of TOV observations that the 'rmse' 
        #is greater than or equal to. The inverse of this would be finding the 
        #quantile (i.e. the TOV value that a given percentage of TOV observations 
        #fall at or below (e.g. quantile = df.quantile(percentile/100) = rmse
print(f'The rmse is greater than or equal to the {percentile:.3f} percentile of TOV observations ({percentile*0.01*len(df1)} out of {len(df1)} observations)')

    #3) R2 (measures how well the best-fit curve fits data)
print(f" The R_squared is: {r2_score(y_test1, y_pred_test1):.3f}")


################# Findings and comments 

# The coefficients(2PA, FTA, AST, Pos_C, Pos_PF, Pos_PG, Pos_SF, Pos_SG) are [0.047, 0.121, 0.167, 0.223, 0.132, 0.147, 0.055, 0.045]
# The intercept is 0.602


#For the predicted v test data the following metrics were found: 
    #1)The mean absolute error has reduced from 0.405 to 0.399
    #2)The root mean squared error has reduced from 0.529 to 0.525
    #3)The R_squared has increased from 0.464 to 0.473 for testing data set

#All chosen variables are significant (P>|t|< 0.05)

#The residuals are not normally distributed but a lot of data is used of this is unlikely:
    #P(omnibus) = 0 (for normal dist =1)
    #P(JB) = 0 (for normal dist >0.05)

#Residuals are not autocorrelated Durbin-Watson = 2.046

#Residual has a slight positive skew (=0.823)
#Residual is also displays some leptokurtic(kurtosis = 6.028 taller and thinner than normal dist curve)

#Cond. No. = 7.67e+16 (note: this is high but is to be expected because of our dummy variables )
##########################################################################################



#############################Making Predictions ###############################

Two_PA = 9.85987 #number of two-point shots taken 
Assists = 6.99363 #number of asists 
FTA = 4.70064
Position = 'Pos_PG' # player position chose one of: Pos_C, Pos_PF, Pos_PG, Pos_SF, Pos_SG

Turnover = intercept1 + coefficients1['AST']*Assists + coefficients1['FTA']*FTA  + coefficients1['2PA']*Two_PA + coefficients1[Position]*1


print(f"A {Position.replace('Pos_', '')} that takes {Two_PA} two-point shots, and makes {Assists} assists may commit {Turnover:.1f} turnovers")

###############################################################################



##### Residual 
# Checking residuals (training dataset)
y_pred1 = result1.predict(X_train1) 

residual_df = y_train1-y_pred1
plt.figure()
plt.scatter(y_pred1,residual_df, marker = '+' )
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

df_QQ = pd.DataFrame(residual_df.sample(frac = 0.10).values, columns = ['Residual'])
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




#############################Saving Model ####################################
# import pickle

#     #stats model 
# with open('Linear_Regrssion_Model_with_Pos_SM.sav','wb') as f:
#       pickle.dump(result1,f)
   
#     #sklearn model 
# with open('Linear_Regrssion_Model_with_Pos_sk.sav','wb') as f:
#       pickle.dump(lm,f)

#############################################################################

################################loading model #################################
# with open('Linear_Regrssion_Model_Numeric_columns_sk.sav','rb') as f:
#       lm = pickle.load(f)

    #r_squared stuff
# r_squared = lm.score(X_test1.values.reshape(-1,9), y_test1.values.reshape(-1,1))

# adjusted_r_squared = 1 - (1-r_squared)*(len(y_test1)-1)/(len(y_test1)-X_test1.shape[1]-2)
