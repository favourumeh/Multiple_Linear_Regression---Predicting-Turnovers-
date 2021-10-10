# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 12:56:39 2021

@author: favou
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math 

from copy import deepcopy

df = pd.read_excel('1990_2021_NBA_ppg_stats.xlsx')

#df1 =  df[['Pos', 'Age', 'MP', '2PA', '3PA', 'FTA', 'TRB','AST', 'TOV', 'PF']]
#df1 =df[['Pos', 'Age', 'MP', '2PA', '2P_per', '3PA', '3P_per','FTA', 'FT_per','TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']] 
df1 =df[['Pos', 'Age', 'MP', '2P', '2PA', '2P_per', '3P','3PA', '3P_per', 'FT', 'FTA', 'FT_per','TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']] 


#Selecting Players that have played > 10mins 
    #note: this is done because this will be changed to per 36 min stats and
           #players with low minutes can have grossly exagerated large statistics 
df1 = df1[df1["MP"]>10] # -- 12841 rows
df1.index = range(len(df1.index))

##################### Exploratory data analysis ###############################
#checking the feature types 
df1.dtypes

#checking for no. of nulls in each column 
df1.isnull().sum() # -- no nulls found

df2 = pd.DataFrame()
df2['Pos'] = df1['Pos']
df2['Age'] = df1['Age']
df2['2P_per']= df1['2P_per']
df2['3P_per']= df1['3P_per']
df2['FT_per']= df1['FT_per']


#Per_game statistics ----> per 36 statistsics 
for c, column in enumerate(df1):
    if column not in ('Pos', 'Age', '2P_per', '3P_per', 'FT_per'):    
        df2[column] = (36*df1[column])/df1['MP']

df2 = df2[['Pos', 'Age', 'MP', '2P', '2PA', '2P_per', '3P', '3PA', '3P_per', 'FT','FTA', 'FT_per', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']] 
#df2 = df2[['Pos', 'Age', 'MP', '2PA', '2P_per', '3PA', '3P_per','FTA', 'FT_per', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']] 


#checking for nulls
df2.isnull().any()
df2.isnull().sum() #number of column that has null values 
df2.isnull().sum()/(df2.shape[0]*0.01) #percentatge of column that has null values 

    
#Handling Nulls
    #isolating null indices
check_null = df2[df2['2P_per'].isnull() | df2['FT_per'].isnull() | df2['3P_per'].isnull()]
null_indicies = check_null.index


    #Checking if the Null are the result of 0 shot attempt (THEY ARE) 
check_null[check_null['3PA'] == 0]
check_null[check_null['2PA'] == 0]
check_null[check_null['FTA'] == 0]

    #Method 1: Simply dropping Nulls 
    
df2.drop(null_indicies, axis =0, inplace = True)
df2.index = range(len(df2.index))
        #checking method outcome
df2.isnull().any()

    #Method 2: Setting the null to 0 
    
df2['2P_per'][(check_null[check_null['2PA'] == 0]).index] = 0
df2['3P_per'] = df2['3P_per'].apply(lambda x: 0 if math.isnan(x) else x)
df2['FT_per'] = df2['FT_per'].apply(lambda x: 0 if math.isnan(x) else x)
        #checking method outcome
df2.isnull().any()
        #logic: if you aren't attempting shots then you are probably bad at it
                #this makes sense for 3P but not FT or 2P

    #Method 3: setting the null values to the medium

df2['2P_per'].fillna(df2['2P_per'][df2['2P_per'].notnull()].median(), inplace = True)
df2['3P_per'].fillna(df2['3P_per'][df2['3P_per'].notnull()].median(), inplace = True)
df2['FT_per'].fillna(df2['FT_per'][df2['FT_per'].notnull()].median(), inplace = True)

df2['2PA'] = df2['2PA'].apply(lambda x: df2['2PA'][df2['2PA'] !=0].median() if x ==0 else x)
df2['3PA'] = df2['3PA'].apply(lambda x: df2['3PA'][df2['3PA'] !=0].median() if x ==0 else x)
df2['FTA'] = df2['FTA'].apply(lambda x: df2['FTA'][df2['FTA'] !=0].median() if x ==0 else x)
        #checking method outcome          
df2.isnull().any()
df2[df2['3PA']==0]
df2[df2['2PA']==0]
df2[df2['FTA']==0]

    #Method 4: setting null values to the medium of the player's position 
#df2A = deepcopy(df2)
#df2 = deepcopy(df2A)

for position in df2.Pos.unique():

    
    for i in [['3PA', '3P', '3P_per'], ['2PA', '2P', '2P_per'], ['FTA', 'FT', 'FT_per']]:
        
        #index for '3PA', '2PA' or 'FTA' equal 0 (for a specific player position)
        index = df2[(df2[i[0]] == 0) & (df2.Pos == position)].index # players no attempts (for ... )
        index2 = df2[(df2[i[0]] != 0) & (df2.Pos == position)].index # remaining players (for ...)
        #mean for 3PA, ... for >0 attempts (for a spec....)
        df2.loc[index, i[0]] = df2.loc[index2, i[0]].mean()  # df2[i[0]][(df2['Pos'] == position)].mean()
        
        #mean for 3P, ... for >0 attempts (for a spec....)
        df2.loc[index, i[1]] = df2.loc[index2, i[1]].mean() # df2[i[1]][df2.Pos == position].mean()
        
        #mean for 3P_per, ... for >0 attempts (for a spec....)
        df2.loc[index, i[2]] = df2.loc[index, i[1]]/df2.loc[index, i[0]]



#investigating uniqueness of features 
for i, column in enumerate(df2):
    unique_val = np.unique(df2[column])
    count_unique = len(unique_val)
    if count_unique < 10:
        print(f" For column {column} the number of unique features are {count_unique}, {unique_val}")
    else:
        print(f" For column {column} the number of unique features are {count_unique}")
        
      # there are 5 positions categories for the 'Pos' column: ['C' 'PF' 'PG' 'SF' 'SG']


#initial data visualisation: pairplot 
g = sns.pairplot(df2)
   
#using correlating to identify which any multicollinearities and variable that don't correlate with 
hm = df2.drop(['Pos','MP', '2P', '3P', 'FT'],axis =1).corr()

        #visualising correlation plot with heat map

g = sns.heatmap(hm, cmap="YlGnBu", annot= True, fmt = '.1g', annot_kws = {'size':8}).set(title = 'Correlation plot of all independent and dependent variables')
        
#g = sns.heatmap(hm, annot= True, annot_kws = {'size':8}).set(title= f"For positions: {' '.join(df2.drop(['Pos'],axis =1).columns)}")

df2B = df2.drop(['MP','2P_per', 'FT_per', 'TRB', 'BLK', 'PF','2P', '3P', 'FT', 'Age', '3PA', '3P_per', 'STL'], axis =1) 

hm = df2B.drop(['Pos'],axis =1).corr()

g = sns.heatmap(hm, cmap="YlGnBu", annot= True, fmt = '.1g', annot_kws = {'size':8}).set(title = 'Correlation plot of all independent and dependent variables')


##Variance Inflation factor
    #This takes one predictor(independent) variable,i, and regresses it against all other 
    #predictor variables. The VIF for a predictor variable,i,  is VIF_i 
    #VIF_i= 1/(1-R2_i) R2 = R-squared, 
    
    #Rule of thumb for VIF:
        # 1 = not correlated.
        # Between 1 and 5 = moderately correlated.
        # Greater than 5 = highly correlated.

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = df2B.drop(['TOV', 'Pos'], axis =1)
X1 = add_constant(X) #add a constant term even though it isn't a predictor because statsmodel requires it 

pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)

    #All VIF are low (<5) so no need to drop anything 


#dropping 'Age', '3PA', 'TRB', 'PF'
df3 = df2.drop(['Age', 'MP', '3PA', '3P_per','2P_per', 'FT_per','TRB', 'PF', 'STL', 'BLK', '3P', '2P', 'FT'], axis =1)

hm = df3.drop(['Pos'],axis =1).corr()

        #visualising correlation plot with heat map
g = sns.heatmap(hm, cmap="YlGnBu", annot= True, fmt = '.1g', annot_kws = {'size':8}).set(title= f"For positions: {' '.join(df3.drop(['Pos'],axis =1).columns)}")



#Pairplot 2: 
g = sns.pairplot(df3)
df3.plot(kind = 'box')

#Ensuring features at least  >0; Can predict these with any kind of linear/polynomial fit 
for column in df3:
    if column != 'Pos':
        df3 = df3.drop(df3[df3[column]== 0].index, axis =0) # --removed 176 rows 

g = sns.pairplot(df3)

#removing outliers 
df4 = df3[(df3.TOV> df3.TOV.quantile(0.0005)) & (df3.TOV< df3.TOV.quantile(0.9995)) & (df3.AST !=0)]

plt.figure()
g = sns.pairplot(df4)

# prob_102PA_1tov = len(df3[ (df3['2PA']>10) & (df3.TOV<1) ])/len(df3)


###########################Visualisng Categoric Data ##########################
df4.Pos.value_counts().plot(kind = 'bar', color = {'blue','purple','orange','green', 'red' })
plt.xlabel('Player Positions')
plt.ylabel('No. of occurences')


#lm plot 1: visualising data by position:
plt.figure()
g = sns.lmplot(x ='AST', y = 'TOV', data = df4, col = 'Pos', ci = False, col_wrap = 3, height = 5, scatter_kws = {'color':'purple'}, line_kws= {'color':'orange'})
plt.xlim(0, 15)
plt.ylim(0, 8)

plt.figure()
g = sns.lmplot(x ='2PA', y = 'TOV', data = df4, col = 'Pos', ci = False, col_wrap = 3, height = 5, scatter_kws = {'color':'purple'}, line_kws= {'color':'orange'})
plt.xlim(0, 22.5)
plt.ylim(0, 8)

plt.figure()
g = sns.lmplot(x ='FTA', y = 'TOV', data = df4, col = 'Pos', ci = False, col_wrap = 3, height = 5, scatter_kws = {'color':'purple'}, line_kws= {'color':'orange'})
plt.xlim(0, 22.5)
plt.ylim(0, 8)

#Box plot 1: Data cleansing: Removing outliers
    #visualising data by player position: boxplot
    
pal = {'SG':'green', 'SF':'red', 'PF':'blue', 'PG':'purple', 'C':'orange'} # palette = pal

plt.figure()
g = sns.boxplot(x = 'Pos', y = 'TOV', data = df4)

plt.figure()
g = sns.boxplot(x = 'Pos', y = 'AST', data = df4)

plt.figure()
g = sns.boxplot(x = 'Pos', y = '2PA', data = df4)

plt.figure()
g = sns.boxplot(x = 'Pos', y = 'FTA', data = df4)

df4.loc[df4[df4.Pos == 'PF'].index, :].median()
df4.loc[df4[df4.Pos == 'PG'].index, :].median()

Describe = df4.describe()

##Determining if Player position affect TOV in ways not explained by 2PA, FTA, AST
import scipy.stats as stats
columns = ['features', 'PF', 'C', 'SG', 'SF', 'PG' ]
mean_std_df = pd.DataFrame() 

column = 'TOV'
mean_std = []
plt.figure()
for position in df4.Pos.unique(): #['PF', 'C', 'SG', 'SF', 'PG']           
    mu = df4[column][df4.Pos == position].mean() 
    sigma = df4[column][df4.Pos == position].std()
    
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label = f'{position}') 
    plt.title(f'The equivalent gaussian (normal) distribution of {column} across different player position ')
    plt.legend()
    plt.xlabel(f'{column}')
    plt.ylabel('Probability(pdf)')
    mean_std.append([mu, sigma])
    
mean_std.insert(0,column)
series = pd.Series(mean_std, index = columns)
mean_std_df = mean_std_df.append(series, ignore_index = True)   

for position in df4.Pos.unique(): #['PF', 'C', 'SG', 'SF', 'PG']           
    # plt.figure()
    df4[df4.Pos == position].loc[:,'TOV'].hist(label = f'{position}', grid = False)
    plt.legend()
    #plt.title(f'TOV distribution for player with position {position}')

###################Visualising changes #######################################

#Box plot 2: 
    #visualising data by player position: boxplot
    
pal = {'SG':'green', 'SF':'red', 'PF':'blue', 'PG':'purple', 'C':'orange'} # palette = pal

plt.figure()
g = sns.boxplot(x = 'Pos', y = 'TOV', data = df4)


#lm plot 2: visualising data by position:

g = sns.lmplot(x ='AST', y = 'TOV', data = df4, col = 'Pos', ci = False, col_wrap = 3, height = 5, scatter_kws = {'color':'green'})

g = sns.lmplot(x ='2PA', y = 'TOV', data = df4, col = 'Pos', ci = False, col_wrap = 3, height = 5, scatter_kws = {'color':'green'})

g = sns.pairplot(df4)

#Saving Data 
#df4.to_csv("Nba_Regression_data.csv", index = False)

# df4.drop('Pos', axis =1).to_csv("Nba_Regression_data_new.csv", index = False)
#df4.to_csv("Nba_Regression_data_new1.csv", index = False)


#############################################
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 

df4A = df4._get_numeric_data()
X = df4A.drop(['TOV'], axis =1)
X_sm = sm.add_constant(X)
y = df4A['TOV']

X_train, X_test, y_train, y_test = train_test_split(X_sm, y, test_size = 0.2, random_state = 0)

result = sm.OLS(y_train, X_train).fit()
result.summary()


##################finding gradients ##########################

#position = 'SG'
variable = 'FTA'

for position in df.Pos.unique():
    df_pl = df[df.Pos == position]
    
    slope, intercept, r_value, p_value, std_err  = stats.linregress(df_pl[variable], df_pl.TOV)
    
    plt.figure()
    ax = sns.regplot(x=variable, y="TOV", ci = False, data=df_pl, color='purple', 
     line_kws={'label':"y={0:.2f}x+{1:.1f}".format(slope,intercept), 'color':'orange'})
    
    plt.title(f'position = {position}')
    plt.legend()