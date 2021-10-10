# Multiple_Linear_Regression-- Predicting Turnovers 

## Project Overview
 - In this project two multiple linear regression models were created to predict the turnovers commited by an NBA player per 36 minutes (TOV)
 - The model results were validated:
 
        - against the linear regression assumptions (Homoscedasticity, No Autocorrelation, Normal Residuals, No Multicollinearity, Exogeneity)
        
        - using the following metrics: 1) Mean Absolute Error (MAE); 2) Root Mean Squared Error (RMSE); 3) R-squared (R2)
        
        - using train-test split and cross-validation 
        
 - Ultimately its was found that Two-point attempts per36(2PA), Free throw Attempts  per36(FTA) and Assists per36(AST)  were the only factors from the dataset that had a notable affected the TOV:
 
        - **R2 = 0.46** . This is low because the model does not factor a player's skill which is the primary predicator of TOV. The R2 suggests that there is a significant variance in skill level amongs players  
        
        - **RMSE = 0.529** . This error is roughly 25% of the mean TOV observed. 

- The models created can be used to assists NBA coaches in player development because the model acts as a predictor of a player's expected TOV. A player vastly outperforming their expected TOV (i.e. commits less TOV that expected) could indicated high skill level which would mean that they require more on-ball possessions.

- For more indepth analysis see the report ('Multiple Linear Regression_with_FTA.docx'). 

## Python version and packages 
Python Version: 3.8.3

Packages: pandas, numpy, sklearn, statsmodles, matplotlib, seaborn, pickle

## Data cleaning, transformation and Feature engineering
- Early feature selection and engineering

- Nulls

- correlation plots
## Exploratory Data Analysis
![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Bar%20chart%20for%20player%20position%20population.png)

![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Correlation%20plot.png)

![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/lmplot%20for%20TOV%20v%20AST.png)

## Model Building 
The dependent variable was TOV
For model 1: Independent variables = 2PA, AST, FTA

For Model 2: Independent variables = 2PA, AST, FTA, Pos (Player Position)

Pos is a categorical variable, and it was added because I believed that Player position would introduce player skill to the model. I hypothesised that the position a player plays in would lend them to cetain skillsets which would impact their likelihood of commiting turnovers. 

Pos was converted into a dummy variable before any modelling.

I split the data into 80-20 train and tests split. 

I used Multiple Linear Regresssion for both models and evaluated them on the test set using the following metrics: Root Mean Squared, Mean Absolute Error and R-squared. I chose MAE as a baseline error metric and RMSE because it punishes extrememely poor predictions. I chose R2 to gauge the how well the addition of the Pos variable explains the variance in player skill.  

To validate the metric results from the test dataset each metric was recalculated with 5-fold cross validation. 


## Model Results and Performance 
Here are the summary results for both models 
![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Model%20Summary%20result.png)

The figure shows the Predicted V Actual turnovers commited per 36minutes 
![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Model%201--%20Predicted%20V%20Actual%20TOV.png)

## Try out Model 
Please feel free to try out the model 
