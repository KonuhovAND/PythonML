from scipy.constants import alpha
from pylab import dtype
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('./datasets/housing/housing.csv',sep=',')
df= pd.DataFrame(data)

#amount of each catagory
print(df['ocean_proximity'].value_counts())
print(df['ocean_proximity'][:5])

# Name: count, dtype: int64
# 0    NEAR BAY
# 1    NEAR BAY
# 2    NEAR BAY
# 3    NEAR BAY
# 4    NEAR BAY
# Name: ocean_proximity, dtype: str

# df.hist(bins=50,figsize=(2,2))
#histogram

#-----------------#
#spliting training set(only one ), receive train_set(origin),and test_st(data to train)
train_set,test_st = train_test_split(df,random_state=42,test_size=.5)

# Make a cut of data , make them like slices in dif catagagoriecs(chip,mid,exp etc)
df['income_cat'] = pd.cut(df['median_income'],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])

# make a 10 test data(ids)
srt = StratifiedShuffleSplit(10,test_size=0.4,random_state=42)

str2 = []
#add data(not ids), due to we pass ids in iloc, which give data according id
for t1, t2 in srt.split(df,df['income_cat']):
    str2.append([
        df.iloc[t1],
        df.iloc[t2]
    ])

    
# receive data from datasets
strat_train_set, strat_test_set = str2[0] 

for set_ in(strat_train_set,strat_test_set): set_.drop('income_cat',axis=1,inplace=True)

housing = strat_train_set.copy()

# print plot
housing.plot(kind='scatter',x='longitude',y='latitude',grid=True,alpha=.2)


plt.show()
