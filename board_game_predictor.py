import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Loading the data.
games = pd.read_csv("games.csv")
#games = games.sample(frac=0.1,random_state=1)

# Printing the names of colums
print("Features ::: ")
print(games.columns)
print("Shape : ",games.shape)

# Print the first row of all games with 0 scores.
print(games[games["average_rating"]==0].iloc[0])

# Print the first row of all games with >0 scores.
print(games[games["average_rating"] > 0].iloc[0])

# Remove any rows without user reviews
games = games[games["users_rated"]>0]

#Remove any rows with missing values.
games = games.dropna(axis=0)

#Make histogram of all average ratings.
plt.hist(games['average_rating'])
plt.title("Average Rating")
plt.xlabel("Rating")
plt.ylabel("Number of Users")
plt.show()

# Feature : 
#['id', 'type', 'name', 'yearpublished', 'minplayers', 'maxplayers',
#       'playingtime', 'minplaytime', 'maxplaytime', 'minage', 'users_rated',
#       'average_rating', 'bayes_average_rating', 'total_owners',
#       'total_traders', 'total_wanters', 'total_wishers', 'total_comments',
#       'total_weights', 'average_weight']
# Note here that, feature 'id' is not at all giving any information about games.
# Using 'id' to analyse the data will overfit the data.

# Correlation metrics and heat map.
corrmat = games.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.title("Correlation Heatmap")
plt.show()

# Get all columns
columns = games.columns.tolist()

# Filter columns to remove data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]

#Store the variable we will be predicting on
target = "average_rating"

# Generate train and test dataset.
train = games.sample(frac=0.8,random_state=1)

# Select anything not in training state and put in test
test = games.loc[~games.index.isin(train.index)]

# Print Shape
print("Train Data Shape : ",train.shape)
print("Test Data Shape : ",test.shape)

#----------------Linear regression Model-------------#

print(end="\n")
print(end="\n")
print(end="\n")
print("Implementation of Linear Regression Model")
# Initialize the model class.
LR = LinearRegression()

# Fit the model training data
LR.fit(train[columns],train[target])

# Generate predictions for the test set
predictions = LR.predict(test[columns])

# Error between predictions and actual value
print("Mean Square Error of Predictions : ",mean_squared_error(predictions,test[target]))

#-----------------------------------------------------#

#----------------Random Forest regressor-------------#

print(end="\n")
print(end="\n")
print(end="\n")
print("Implementation of Random Forest Regressor")
# Initialize model
RFR = RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)

# Fit to the data
RFR.fit(train[columns],train[target])

# Make Predictions
predictions = RFR.predict(test[columns])

# Compute error of Predictions
print("Mean Square Error of Predictions : ",mean_squared_error(predictions,test[target]))

#----------------------------------------------------#

#-------------------Sample Prediction----------------#

print(end="\n")
print(end="\n")
print(end="\n")
print("Sample prediction for first game")
# Making predictions with both models for first game
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1,-1))

print("Actual Rating value of first game : ",test[target].iloc[0])
print("Rating value predicted by Linear Regression : ",rating_LR)
print("Rating value predicted by Random Forest Regressor : ",rating_RFR)

#----------------------------------------------------#




