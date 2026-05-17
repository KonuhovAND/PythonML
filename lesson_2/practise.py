import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'],dtype=int)
s = s+10
print(s)
print('-' * 30)
print(f"s.min = {s.max()},\ns.max = {s.max()}, \ns.sum = {s.sum()},\ns.mean() = {s.mean()}")
print('-' * 30)


data = range(0,10,2)
step = len(data)

a = pd.Series(data= data, index=np.arange(0,step))
b = pd.Series(data, index=np.arange(step - step // 2, 2 * step - step // 2))
print(a,b)
print('-' * 30)
print(a+b)

print('-' * 30)

data = [
 { 'index': 'KZ', 'country': 'Kazakhstan', 'population': 18.04, 'square': 2724902 },
 { 'index': 'RU', 'country': 'Russia', 'population': 144.5, 'square': 17125191 },
 { 'index': 'BY', 'country': 'Belarus', 'population': 9.5, 'square': 207600 },
 { 'index': 'UA', 'country': 'Ukraine', 'population': 45.5, 'square': 603628 },
]

data_frame= pd.DataFrame(data)
print(data_frame)
data_frame.set_index("index",inplace=True)
data_frame.head()
print('-' * 30)
print(data_frame.loc['KZ'])
print('-' * 30)
print(data_frame.iloc[1])

print('-' * 30)
data_frame["nigga_count"] = f"{np.random.randint(-10,10)}"+ data_frame['country']

print(data_frame)
data_frame.drop("nigga_count",inplace=True,axis=1)
print(data_frame)
print()



print('-' * 30)
data_frame = pd.read_csv("data.csv",sep=',')
print(data_frame.head(5))
print('-' * 30)

print(data_frame.sort_values(['Make','MPG_City'],ascending=True).head())

print('-' * 30)
print(pd.crosstab(data_frame["Model"],data_frame['Make']))

plt.figure(figsize=(8, 3))
sns.heatmap(pd.crosstab(data_frame['Model'], data_frame['EngineSize']), cmap="jet", annot=False, cbar=True)
plt.show()