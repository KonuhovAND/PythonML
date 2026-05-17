import pandas as pd
import numpy as np

# ================= ЗАДАЧА 4: Агрегация (1 балл) =================
# Посчитайте среднюю зарплату и максимальный возраст
data = {'name': ['Anna', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'salary': [50000, 60000, 75000]}
df = pd.DataFrame(data)

result = df['salary'].mean()
print(result)
# Ожидаемый вывод: (61666.666666666664, 35)

print('-' * 30)

# ================= ЗАДАЧА 5: groupby (2 балла) =================
df_grp = pd.DataFrame({
    'city': ['Moscow', 'Moscow', 'SPb', 'SPb', 'Moscow'],
    'sales': [100, 150, 200, 50, 300]
})

sales_sum = df_grp.groupby(['city']).sum()
print(sales_sum)
print('-' * 30)
# Ожидаемый вывод:
# city
# Moscow    550
# SPb       250
# Name: sales, dtype: int64


# ================= ЗАДАЧА 6: Merge (2 балла) =================
left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value_left': [1, 2, 3]})
right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value_right': [4, 5, 6]})

merged =pd.merge(left,right,on='key',how='left') 
print(merged)
print('-' * 30)
# Ожидаемый вывод:
#   key  value_left  value_right
# 0   A           1          4.0
# 1   B           2          5.0
# 2   C           3          NaN


# ================= ЗАДАЧА 7*: Категории (бонус +1) =================
df_grp['city'] = df_grp['city'].astype("category")
df_grp['sales']= df_grp['sales'].astype("int32")
df_grp.info()
# Ожидаемый вывод: в конце info() memory usage должно уменьшиться
print('-' * 30)