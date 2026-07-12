import pandas as  pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

start = dt.datetime.now()

data = pd.read_csv('D:\\Code\\PythonML\\book\\housing.csv',sep=',')
preprocess = Pipeline([
    ('prep',ColumnTransformer([
        ('num',Pipeline([
            ('imu',SimpleImputer(strategy='median')),
            ('sc',StandardScaler,)
        ]),['longitude', 'latitude', 'housing_median_age', 
             'total_rooms', 'total_bedrooms', 'population', 
             'households', 'median_income']),
        ('cat',OneHotEncoder,['ocean_proximity'])
    ])),
    ('reg',RandomForestRegressor(random_state=42))
])

housing = data.drop("median_house_value",axis=1)
housing_labels =  data['median_house_value'].copy()

X_tr,x_ts,y_tr,y_ts = train_test_split(housing,housing_labels,test_size=.4,random_state=42)
housing_preprocessed = preprocess.fit(X_tr,y_tr)
y_prd = preprocess.predict(x_ts)

rmse = root_mean_squared_error(y_pred=y_prd,y_true=y_ts)
print(rmse)

print(dt.datetime.now() - start)