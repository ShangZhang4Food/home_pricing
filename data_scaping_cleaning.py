# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 20:20:38 2021

@author: jerry
"""

# import opendatasets as od
import pandas as pd
import os

############# Step 1. Extract data from Kaggle and import the data ####################

path = os.getcwd()
# Import the dataset from Kaggle
# od.download("https://www.kaggle.com/c/house-prices-advanced-regression-techniques")

# Read the training and testing datasets from csv files
# train_df = pd.read_csv("house-prices-advanced-regression-techniques/train.csv", sep = ',')
### train the csv with removed outliers.
train_df = pd.read_csv("train_removeO.csv", sep = ',')
test_df = pd.read_csv("house-prices-advanced-regression-techniques/test.csv", sep = ',')



############# Step 2. Dealing with missing values ####################


# Convert the categorical variables into dummy variables
train_df = pd.get_dummies(train_df, dummy_na=False)
test_df = pd.get_dummies(train_df, dummy_na=False)

# There are still three variables have missing values:
# Deal with LotFrontage - filled with 0. 
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(0)

# Deal with GarageYrBlt - Delete (Garage year built is probably not important)
del train_df['GarageYrBlt']
del test_df['GarageYrBlt']

# Delete ID column
del train_df['Id']
del test_df['Id']

# Deal with MasVnrArea
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(0)

# Move the dependent variable to the end of the dataframe
train_df = train_df[[c for c in train_df if c not in ['SalePrice']] 
       + ['SalePrice']]

train_df.to_csv(path + '\\train_cleaned.csv', index = False)
test_df.to_csv(path + '\\test_cleaned.csv', index = False)






