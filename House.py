import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
df = pd.read_csv('Projects/data.csv')

# Define features and target
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition','sqft_above','sqft_basement','yr_built','yr_renovated','city']
target = 'price'

# One-hot encode the 'city' column
df_encoded = pd.get_dummies(df, columns=['city'], drop_first=True)

# Define X and y
X = df_encoded.drop(columns=[target,'date','street','statezip','country'])
y = df_encoded[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Regressor
dtree = DecisionTreeRegressor()
dtree = dtree.fit(X_train, y_train)

# Prepare the input data for prediction
input_data = pd.DataFrame([[4,1.5,1500,8000,2,0,4,5,1500,0,2000,2005,'Seattle']], columns=features)

# One-hot encode the input data
input_data_encoded = pd.get_dummies(input_data, columns=['city'], drop_first=True)

# Align the columns of the input data with the training data
input_data_encoded = input_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make the prediction
prediction = dtree.predict(input_data_encoded)

# Print the prediction
print(prediction[0])
