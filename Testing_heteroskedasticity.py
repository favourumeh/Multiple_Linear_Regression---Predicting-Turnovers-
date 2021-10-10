# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 18:24:06 2021

@author: favou
"""

def Breushpagan_test(result):
    #This function determines if a regression model displays heteroscedasticity 
    #It takes in the fitted regression model given that you have used 
    #'statsmodels.api' to create your model. It then conducts the Breush-Pagan 
    #test to heteroscedasticity at 95% confidence interval
  
    from statsmodels.compat import lzip
    import statsmodels.stats.api as sms


    names = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
    test = sms.het_breuschpagan(result.resid, result.model.exog)

    
    het_pvalue = lzip(names, test)[1][1]
    
    if het_pvalue <0.05:
        print("At 95% confidence interval NO heteroscedasticity present in model")
    else:
        print("At 95% confidence interval heteroscedasticity present in model")

    return print(f" p-value = {het_pvalue:.3f}")