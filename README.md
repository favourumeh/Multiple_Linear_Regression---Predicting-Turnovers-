# Multiple_Linear_Regression-- Predicting Turnovers 

- For more in-depth analysis see the report ('Multiple Linear Regression_with_FTA.docx'). Otherwise, the rest of the README gives a summary of some of the actions undergone and the findings of the project. 

- Please feel free to try out the models using the 'Try_out_Model.py' file in the 'Try Out Models' folder

## Project Overview
 - In this project two multiple linear regression models were created to predict the turnovers committed by an NBA player per 36 minutes (TOV)
 
 - The model results were validated against the linear regression assumptions and using: 
        
      - the following metrics: 1) Mean Absolute Error (MAE); 2) Root Mean Squared Error (RMSE); 3) R-squared (R2)
        
      - train-test split and cross-validation 
        
 - Ultimately it was found that Two-point attempts per36(2PA), Free throw Attempts per36(FTA) and Assists per36(AST) were the only factors from the dataset that had a notable effect on the TOV:
 
      - **R2 = 0.46** . This is low because the model does not factor a player's skill which is the primary predictor of TOV. The R2 suggests that there is a significant variance in skill level amongst players  
        
      - **RMSE = 0.529** . The value of this error is roughly 25% of the mean TOV observed. 

- The models created can be used to assists NBA coaches in player development because the model acts as a predictor of a player's expected TOV. A player vastly outperforming their expected TOV (i.e. commits less TOV than expected) could indicate good ball retention skills which would mean that they require more on-ball possessions. It can also be used to pinpoint players who need more training in ball retention. 


## Python version and packages 
Python Version: 3.8.3

Packages: pandas, numpy, sklearn, statsmodles, matplotlib, seaborn, pickle

## Data used 

Slice of raw data table used. 

![alt text](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/slice%20of%20data.png)

 - These are per game statictics and they were converted to per 36 minutes during the preprocessing phase 
 - Teams(Tm), Games played(G) and Games started(GS) and Year were removed from the data as they were irrelevant to the dependent variable TOV

Glossary of table headings

 ![](https://github.com/favourumeh/Identifying-Player-Position/blob/main/images%20dump/feature%20definitions.png)
 
 - Pos is  player position. There are 5 positions: 
    - 'C' = Centre
    - 'PF' = Power Forward
    - 'PG' = Point Guard
    - 'SF' = Small Forward
    - 'SG' = Shooting Guard
 - MP is minutes played per game
- For further context behind the variables see the 'Features_Explained.docx' file 

## Data cleaning, transformation and Feature engineering
- Nulls

    - All instances of Null values were found in the percentages columns (i.e. 2P_per, 3P_per and FT_per) and the cause of this was division by zero. All null elements were due to a player not attempting a particular shot (i.e 2PA, 3PA, FTA = 0). Therefore 2P_per = 2P/2PA = nan.  
    
    - Instead of removing the rows that featured nulls the average shot attempts and shots made for each player position was calculated and assigned to the rows were 2PA, 3PA or FTA = 0. The percentage columns were then recalculated from this (e.g. 2P_per = 2P/2PA). 
    
    - For example, if a player played the centre position and they had zero 3PA  then the average 3PA and 3P for centres would be assigned to them.
    
- Early feature selection and engineering:

    - Obvious instances of multicollinearity were removed (e.g. The shots made and shot attempted columns are highly correlated so any shots made columns such as 2P was removed)
    
    - 'Minutes played' (MP) highly correlated with most of the other features but it could not be removed without violating Exogeneity assumption, so its effect was dampened by extrapolating all relevant statistics to per 36 minutes statistics 
    
    ![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Feature%20engineering%20and%20selection.png)
    
- correlation heatmaps and variance inflation factors were used to determine the not-so-obvious correlation amongst features 

## Exploratory Data Analysis
Here is the distribution of the player position population used for the final model

![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Bar%20chart%20for%20player%20position%20population.png)

These correlation plots show feature selection

![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Correlation%20plot.png)

These 'lmplot' show how the relationship between TOV and AST changes for different player positions

![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/lmplot%20for%20TOV%20v%20AST.png)

## Model Building 
The dependent variable was TOV
For model 1: Independent variables = 2PA, AST, FTA

For Model 2: Independent variables = 2PA, AST, FTA, Pos (Player Position)

Pos is a categorical variable, and it was added because I believed that Player position would introduce player skill to the model. I hypothesized that the position a player plays in would lend them to certain skillsets which would impact their likelihood of committing turnovers. 

Pos was converted into a dummy variable before any modelling.

I split the data into 80-20 train and tests split. 

I used Multiple Linear Regression for both models and evaluated them on the test set using the following metrics: Root Mean Squared, Mean Absolute Error and R-squared. I chose MAE as a baseline error metric and RMSE because it punishes extremely poor predictions. I chose R2 to gauge how well the addition of the Pos variable explains the variance in player skill.  

To validate the metric results from the test dataset each metric was recalculated with 5-fold cross validation. 


## Model Results and Performance 
Here are the summary results for both models 

![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Model%20Summary%20result.png)

The figure shows the Predicted V Actual turnovers committed per 36minutes 

![](https://github.com/favourumeh/Multiple_Linear_Regression---Predicting-Turnovers-/blob/main/Images/Model%201--%20Predicted%20V%20Actual%20TOV.png)

