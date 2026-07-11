import pandas as  pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,MinMaxScaler
start = dt.datetime.now()

data = pd.read_csv('D:\\Code\\PythonML\\book\\housing.csv',sep=',')

housing = data.drop("median_house_value",axis=1)
housing_labels =  data['median_house_value'].copy()
housing_num = housing.select_dtypes(include=[np.number])
min_max_sc = MinMaxScaler(feature_range=(-1,1))
housing_min_max_sc = min_max_sc.fit_transform(housing_num)

print(dt.datetime.now() - start)