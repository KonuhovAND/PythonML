import pandas as  pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,MinMaxScaler
start = dt.datetime.now()
# asdfasdf
data = pd.read_csv('D:\\Code\\PythonML\\book\\housing.csv',sep=',')

corr_data = data.corr(numeric_only=True)
print(corr_data["median_house_value"].sort_values(ascending= 0))

corr_data = data.corr(numeric_only=True)
data['niggas'] = data['total_rooms'] / data['population']
print(corr_data["median_house_value"].sort_values(ascending= 1))

housing = data.drop("median_house_value",axis=1)
housing_labels =  data['median_house_value'].copy()

imputer = SimpleImputer(strategy='median')
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit_transform(housing_num)

print(imputer.statistics_)
print(housing_num.median().values)

onHotEn = OneHotEncoder()
cat_en = onHotEn.fit_transform(housing[["ocean_proximity"] ] )
print(cat_en.toarray())

print(dt.datetime.now()- start)