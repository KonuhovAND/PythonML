# ==============================================================================
# Machine Learning Practice: KNN & Feature Scaling (Using data.csv)
# For immediate pasting into Neovim (.py file format)
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# ==============================================================================
# task1
# - Load the local dataset 'data.csv' using pandas ensuring 'index_col=False' is set.
# - Define the feature matrix X using only the numerical columns: 
#   ['MSRP', 'Invoice', 'EngineSize', 'Cylinders', 'Horsepower', 'MPG_City', 'MPG_Highway', 'We']
# - Define the target variable y using the categorical column 'DriveTrain'.
# - Split the data into training and validation sets using train_test_split with 
#   test_size=3 and random_state=42.
# ==============================================================================

# Your code here

data= pd.read_csv('data.csv', sep=',')
x = data[['MSRP', 'Invoice', 'EngineSize', 'Cylinders', 'Horsepower', 'MPG_City', 'MPG_Highway', 'Weight']]
y = data['DriveTrain']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=3, random_state=42)

# ==============================================================================
# task2
# - Initialize a baseline KNeighborsClassifier with K = 2 using uniform weights.
# - Fit the model using the raw, unscaled training features from Task 1.
# - Compute and print the validation accuracy score.
# - Apply a StandardScaler to scale your training and validation features to zero mean and unit variance.
# - Fit an identical K = 2 model on the scaled training data, evaluate its validation accuracy, and print both accuracy values side by side to see the effect of scaling.
# ==============================================================================

# Your code here
Knn = KNeighborsClassifier(n_neighbors=2, weights='uniform')
Knn.fit(x_train, y_train)
y_pred = Knn.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy (Unscaled): {accuracy}")

# Apply StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

# Fit the scaled model
Knn_scaled = KNeighborsClassifier(n_neighbors=2, weights='uniform')
Knn_scaled.fit(x_train_scaled, y_train)
y_pred_scaled = Knn_scaled.predict(x_val_scaled)
accuracy_scaled = accuracy_score(y_val, y_pred_scaled)
print(f"Validation Accuracy (Scaled): {accuracy_scaled}")

# ==============================================================================
# task3
# - Implement a loop that iterates through two distinct distance metrics: ['euclidean', 'manhattan'].
# - Within that loop, test two separate weighting strategies:
#   1. weights='uniform' (all neighbors exert an equal vote).
#   2. weights='distance' (voting power scales inversely with distance).
# - Train the model using the scaled features from Task 2 and evaluate performance on the validation set.
# - Track and print the validation score for all combinations to identify which metric-weight pairing performs best.
# ==============================================================================

# Your code here
for i in ['euclidean','manhattan']:
    for j in ['uniform','distance']:
        Knn_scaled=KNeighborsClassifier(n_neighbors=2,metric=i,weights=j)
        Knn_scaled.fit(x_train,y_train)
        y_pred_scaled=Knn_scaled.predict(x_val_scaled)
        accuracy_scaled=accuracy_score(y_val,y_pred_scaled)
        print()
        print(f"Validation Accuracy (Scaled): {accuracy_scaled}")

# ==============================================================================
# task4
# - Set up a grid search loop to evaluate model configuration performance across combinations of hyperparameters.
# - Define a neighbor count search array: neighbors_vals = [1, 2, 3].
# - Write a double-nested loop:
#   - Outer loop: selection between 'euclidean' and 'manhattan' metrics.
#   - Inner loop: values from neighbors_vals.
# - Train a KNeighborsClassifier on the scaled training data for each parameter combination and evaluate it on the validation set.
# - Find and print the optimal combination of K and distance metric that achieves the highest accuracy.
# ==============================================================================

# Your code here
from itertools import product
def my_func(param):
    metric,n_= param
    Knn_scaled=KNeighborsClassifier(n_neighbors=n_,metric=metric)
    Knn_scaled.fit(x_train_scaled,y_train)
    y_pred_scaled=Knn_scaled.predict(x_val_scaled)
    accuracy_scaled=accuracy_score(y_val,y_pred_scaled)
    print()
    print(f"n = {n_}, metric = {metric} Validation Accuracy (Scaled): {accuracy_scaled}")
    
comb = list(product(['euclidean','manhattan'],range(1,4)))
list(map(my_func,comb))
     

# ==============================================================================
# task5
# - Shift to a regression task using the same 'data.csv' dataset.
# - Isolate the continuous column 'Horsepower' as your feature matrix X and treat 'MSRP' as your continuous target y.
# - Split the data using train_test_split with test_size=3 and random_state=42.
# - Initialize and train a KNeighborsRegressor model using K = 2.
# - Generate continuous value predictions on the validation set and compute the Mean Squared Error (MSE).
# ==============================================================================

# Your code here

x =  data[['Horsepower']]
y = data['MSRP']
x_trained,x_val,y_trained,y_val = train_test_split(x,y,test_size=3, random_state=42)

knn = KNeighborsRegressor(n_neighbors=2,weights='uniform')
knn.fit(x_trained,y_trained)
y_pred=knn.predict(x_val)
y_acc = mean_squared_error(y_val,y_pred)
print(y_acc)