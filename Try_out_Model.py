# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:51:44 2021

Try out the models created. Model 1 does not factor player position whilst Model 2 does

@author: favou
"""
import pandas as pd
import pickle 

############################# Change Accordingly ##############################

#Select Model
Mod = 'model 1' # Options 'model 1' and 'model 2'

# Choose Parameters of Model

Two_PA = 9.85987 #number of two-point shots taken 
FTA = 4.70064 #number of free throws attempted
Assists = 6.99363 #number of assists 
    
    #if applicable(i.e. Model 2 only) 
Position = 'Pos_PG' # player position chose one of: Pos_C, Pos_PF, Pos_PG, Pos_SF, Pos_SG

###############################################################################


# Loading model

if Mod == 'model 1':
    filename ='Model1_statsmodels.sav'
else:
    filename = 'Model2_statsmodels.sav'

with open(filename,'rb') as f:
      lm = pickle.load(f)
      
coefficients = lm.params[1:len(lm.params)]
intercept = lm.params[0]
      
a = 0 if Mod == 'model 1' else coefficients[Position]*1
     

# Calculating TOV
Turnover = intercept + coefficients['AST']*Assists + coefficients['2PA']*Two_PA + coefficients['FTA']*FTA + a

if Mod == 'model 1':
    print(f"A player that takes {Two_PA} two-point shots and has  {Assists} assists and {FTA} freethrow attempts may commit {Turnover:.1f} turnovers")

else:
    print(f"A {Position.replace('Pos_', '')} that takes {Two_PA} two-point shots, and makes {Assists} assists may commit {Turnover:.1f} turnovers")
