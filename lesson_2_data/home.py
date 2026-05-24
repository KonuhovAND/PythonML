from matplotlib.pylab import quantile
from numpy.ma import mean
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
SIZE = 10000
PATH_TO_DATASET = "Billionaires Statistics Dataset.csv"

if not os.path.exists(PATH_TO_DATASET):
    print(f"Предупреждение: Не найден файл {PATH_TO_DATASET}.")
    print("Создаем демонстрационный датасет для проверки структуры...")
    np.random.seed(42)
    
    demo_data = {
        "personName": [f"Billionaire {i}" for i in range(1, SIZE + 1)],
        "finalWorth": np.random.randint(1000, 100000, size=SIZE),
        "age": np.random.choice(
            [np.nan, 25, 45, 56, 72, 80, -5, 150], 
            size=SIZE, 
            p=[0.10, 0.16, 0.16, 0.16, 0.16, 0.16, 0.05, 0.05]
        ),
        "category": np.random.choice(
            ["Technology", "Finance", "Fashion", "Energy"], size=SIZE
        ),
        "selfMade": np.random.choice([True, False], size=SIZE),
        "gdp_country": np.random.randint(10**11, 10**13, size=SIZE,dtype=np.uint64),
        "life_expectancy_country": np.random.uniform(60, 85, size=SIZE),
    }
    df = pd.DataFrame(demo_data)
else:
    df = pd.read_csv(PATH_TO_DATASET)
    print("Датасет успешно загружен из CSV-файла!")

print(f"Формат данных: {df.shape[0]} строк, {df.shape[1]} колонок.\n")


# =====================================================================
# ЗАДАЧА 1: Качество данных (Проверка корректности)
#
# Задание:
# 1. Найдите в колонке 'age' аномальные значения (меньше 0 или больше 100).
# 2. Замените обнаруженные аномалии на пропуски (np.nan).
# 3. Заполните все пропуски в колонке 'age' медианным значением возраста.
# =====================================================================
print("--- Задача 1 ---")

# === ВАШ КОД ЗДЕСЬ ===
df.loc[(df['age'] < 0) | (df['age'] > 100), 'age'] = np.nan
print(f"Количество пропусков в 'age' после очистки и заполнения: {df['age'].isna().sum()}")
df['age'] = df['age'].fillna(df['age'].median())
print(f"Медианный возраст: {df['age'].median()}\n")

print(df)
# =====================================================================
# ЗАДАЧА 2: Сводные показатели и группировка
#
# Задание:
# 1. Сгруппируйте данные по признаку 'selfMade'.
# 2. Рассчитайте для каждой группы среднее и медианное значение возраста ('age').
# 3. Рассчитайте для каждой группы среднее значение состояния ('finalWorth').
# =====================================================================
print("--- Задача 2 ---")

# === ВАШ КОД ЗДЕСЬ ===
grouped_stats = None
grouped_stats = df.groupby('selfMade')
print(grouped_stats['age'].agg(['mean','median']) ) 
print(f"{grouped_stats['finalWorth'].agg(['mean'])}")
print("\n")


# =====================================================================
# ЗАДАЧА 3: Линейная корреляция Пирсона
#
# Задание:
# 1. Постройте матрицу линейной корреляции Пирсона между богатством 
#    ('finalWorth') и числовыми признаками: 'age', 'gdp_country', 
#    'life_expectancy_country'.
# =====================================================================
print("--- Задача 3 ---")

columns_to_corr = ["finalWorth", "age", "gdp_country", "life_expectancy_country"]
columns_to_corr = [col for col in columns_to_corr if col in df.columns]


# === ВАШ КОД ЗДЕСЬ ===
corr_matrix = df[['age','finalWorth','gdp_country','life_expectancy_country']].corr(method='pearson')

print("Матрица корреляции:")
print(corr_matrix)
print("\n")


# =====================================================================
# ЗАДАЧА 4: Визуализация одномерных распределений (Matplotlib)
#
# Задание:
# 1. Постройте график "Ящик с усами" (Box-plot) для состояний миллиардеров ('finalWorth').
# 2. Добавьте на график сетку, подпись для оси Y и информативный заголовок.
# =====================================================================
print("--- Задача 4 (Построение графика) ---")

fig, ax = plt.subplots(figsize=(8, 6))

# === ВАШ КОД ЗДЕСЬ ===
# ax.boxplot(df[df['finalWorth'] < df['finalWorth'].quantile(.95)]['finalWorth'])
ax.boxplot(df.loc[(df['finalWorth'] < df['finalWorth'].quantile(.95)),'finalWorth'] )
ax.set_title("")
ax.set_ylabel('FinalWorth')
ax.set_xlabel('Money')

plt.tight_layout()
# plt.show()


# =====================================================================
# ЗАДАЧА 5: Многомерный анализ (Диаграмма рассеяния / Scatter Plot)
#
# Задание:
# 1. Постройте диаграмму рассеяния (Scatter Plot), отображающую 
#    взаимосвязь возраста ('age') по оси X и состояния ('finalWorth') по оси Y.
# 2. Настройте подписи осей X и Y, а также добавьте заголовок графика.
# =====================================================================
print("--- Задача 5 (Построение графика) ---")

fig, ax = plt.subplots(figsize=(100, 6))

# === ВАШ КОД ЗДЕСЬ ===
ax.scatter(df['age'],df['finalWorth'])
ax.set_ylabel("finalWorth")
ax.set_xlabel("age")
ax.set_title("взаимосвязь возраста ('age') по оси X и состояния ('finalWorth') по оси Y.")


plt.tight_layout()
plt.show()

print("\nКод подготовлен к выполнению. Жду ваши решения и результаты на проверку!")
