"""
Глава 2: End-to-End Machine Learning Project
Упражнения 1–7  |  Датасет: California Housing

Инструкция:
- Заполняйте код под каждым комментарием # === ВАШ КОД ЗДЕСЬ ===
- Запускайте файл — увидите результаты своей работы
- Не меняйте структуру, только реализуйте логику
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.linear_model import Ridge

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
print("=" * 60)
print("Загрузка California Housing Dataset...")
print("=" * 60)
# Пробуем загрузить с sklearn
try:
    from sklearn.datasets import fetch_california_housing

    housing_data = fetch_california_housing()
    data = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
    data["MedHouseVal"] = housing_data.target
    print(f"Загружено: {data.shape[0]} строк, {data.shape[1]} колонок\n")
except ImportError:
    print("Не удалось загрузить через sklearn. Создаю демо-датасет...\n")
    np.random.seed(42)
    n = 10000
    data = pd.DataFrame(
        {
            "MedInc": np.random.lognormal(mean=np.log(3), sigma=0.5, size=n),
            "HouseAge": np.random.randint(1, 52, size=n),
            "AveRooms": np.random.lognormal(mean=np.log(5), sigma=0.3, size=n),
            "AveBedrms": np.random.lognormal(mean=np.log(1.5), sigma=0.3, size=n),
            "Population": np.random.randint(100, 10000, size=n),
            "AveOccup": np.random.uniform(0.5, 10, size=n),
            "Latitude": np.random.uniform(32.5, 42.0, size=n),
            "Longitude": np.random.uniform(-124.5, -114.0, size=n),
            "MedHouseVal": np.random.uniform(0.15, 5.0, size=n),
        }
    )

# Разделяем на фичи и цель
housing = data.drop("MedHouseVal", axis=1)
housing_labels = data["MedHouseVal"].copy()

# Базовый сплит
X_train, X_test, y_train, y_test = train_test_split(
    housing, housing_labels, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples\n")


# ============================================================
# УПРАЖНЕНИЕ 1: Support Vector Machine Regressor (SVR)
#
# Попробуйте SVR с разными гиперпараметрами:
# - kernel="linear" с разными C (напр. 0.1, 1, 10, 100)
# - kernel="rbf" с разными C и gamma (напр. scale, auto, 0.1, 1)
#
# ⚠️ SVR плохо масштабируется → работаем на ПЕРВЫХ 5000 строках
#    и используем 3-fold cross-validation.
#
# Вопрос: Какой SVR показал лучший результат? (сравните RMSE)
# ============================================================
print("-" * 60)
print("УПРАЖНЕНИЕ 1: SVR — поиск гиперпараметров")
print("-" * 60)

# Берём только 5000 строк для скорости
X_train_small = X_train.iloc[:5000]
y_train_small = y_train.iloc[:5000]

# === ВАШ КОД ЗДЕСЬ ===
# 1. Создайте Pipeline: StandardScaler → SVR
# 2. Настройте GridSearchCV с param_grid по C, kernel, gamma
# 3. Используйте cv=3, scoring='neg_mean_squared_error'
# 4. Обучите и выведите лучшие параметры и лучший score

# Пример структуры:
# pipeline = Pipeline([...])
# param_grid = {...}
# grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=..., n_jobs=-1)
# grid_search.fit(X_train_small, y_train_small)
# print(f"Лучшие параметры: {grid_search.best_params_}")
# best_rmse = np.sqrt(-grid_search.best_score_)
# print(f"Лучший RMSE (CV): {best_rmse:.4f}")

# print("TODO: Реализовать упражнение 1\n")
# start = dt.datetime.now()
# preprocessing = Pipeline(
#     [
#         (
#             "scal",
#             StandardScaler(),
#         ),
#         ("reg", SVR(kernel="rbf")),
#     ]
# )

# param_grid = {
#     "reg__C": [0.1, 0.5, 1.0],
#     "reg__gamma": [0.5, 1.0, 0.01],
#     "reg__epsilon": [0.1, 0.2],
# }
# grid_search = GridSearchCV(
#     preprocessing, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1
# )

# grid_search.fit(X_train_small, y_train_small)
# print(f"The best params:{grid_search.best_params_}")

# best_rmse = np.sqrt(-grid_search.best_score_)
# bp = grid_search.best_params_
# print(f"The best rmse {best_rmse}")
# print(dt.datetime.now() - start)
# ============================================================
# УПРАЖНЕНИЕ 2: GridSearchCV → RandomizedSearchCV
#
# Замените GridSearchCV из упражнения 1 на RandomizedSearchCV.
# Используйте n_iter=10 или 20 (чтобы перебрать ~10-20 случайных комбинаций).
#
# Вопрос: Нашлись ли параметры лучше, чем в GridSearch? Быстрее ли работало?
# ============================================================
print("-" * 60)
print("УПРАЖНЕНИЕ 2: RandomizedSearchCV вместо GridSearchCV")
print("-" * 60)

# === ВАШ КОД ЗДЕСЬ ===
# 1. Создайте param_distributions (словарь с диапазонами, как в GridSearch)
# 2. RandomizedSearchCV(..., n_iter=15, cv=3, scoring='neg_mean_squared_error', ...)
# 3. Обучите, выведите лучшие параметры и RMSE

# start = dt.datetime.now()
# print("TODO: Реализовать упражнение 2\n")
# RandomizedSearchCV_ = RandomizedSearchCV(
#     preprocessing, param_grid, n_iter=10, scoring="neg_mean_squared_error", cv=3
# )
# RandomizedSearchCV_.fit(X_train_small, y_train_small)
# print(
#     f"The best params: {RandomizedSearchCV_.best_params_}\n\
#     The best rmse: {np.sqrt(-RandomizedSearchCV_.best_score_)}"
# )
# print(dt.datetime.now() - start)
# ============================================================
# УПРАЖНЕНИЕ 3: SelectFromModel в пайплайне
#
# Добавьте SelectFromModel в пайплайн подготовки, чтобы
# оставить только самые важные признаки.
#
# Подсказка: используйте RandomForestRegressor как базовую модель
# для SelectFromModel (max_features выставит автоматически).
# ============================================================
print("-" * 60)
print("УПРАЖНЕНИЕ 3: SelectFromModel + Pipeline")
print("-" * 60)


# === ВАШ КОД ЗДЕСЬ ===
# 1. Создайте Pipeline:
#    StandardScaler → SelectFromModel(RandomForestRegressor(...)) → SVR(kernel="rbf")
# 2. Обучите на X_train_small, y_train_small
# 3. Оцените RMSE на X_test (полные данные, но с тем же пайплайном)
# 4. Выведите: сколько признаков отобрано, какие именно

# print("TODO: Реализовать упражнение 3\n")

# preprocessing_Select_From_Model = Pipeline(
#     [
#         ("scal", StandardScaler()),
#         (
#             "sfm",
#             SelectFromModel(RandomForestRegressor(n_jobs=2)),
#         ),
#         ("reg", SVR(kernel="rbf")),
#     ]
# )
# params = {
#     "reg__C": [0.5, 1.0],
#     "reg__gamma": [0.1],
#     "reg__epsilon": [0.1, 0.5],
# }

# grid_search = GridSearchCV(
#     preprocessing_Select_From_Model,
#     params,
#     cv=3,
#     n_jobs=-1,
# )

# grid_search.fit(X_train_small, y_train_small)

# print(f"The best params: {grid_search.best_params_}")
# print(bp)
# print(f"The best rmse: {np.sqrt(grid_search.best_score_)}")

# y_pred = grid_search.predict(X_test)

# rmse = root_mean_squared_error(y_test, y_pred)
# print(f"{rmse} - real rmse")

# ============================================================
# УПРАЖНЕНИЕ 4: Кастомный трансформер — KNN Regressor
#
# Создайте свой трансформер KNNIncomeTransformer:
# - fit(X, y=None): обучает KNeighborsRegressor на широте/долготе,
#   предсказывая median_income (название фичи — 'MedInc').
# - transform(X): возвращает X с ДОБАВЛЕННОЙ колонкой
#   'knn_income' — предсказанный доход на основе соседних районов.
#
# Потом добавьте этот трансформер в preprocessing pipeline.
# ============================================================
print("-" * 60)
print("УПРАЖНЕНИЕ 4: Кастомный трансформер KNN")
print("-" * 60)


class KNNIncomeTransformer(BaseEstimator, TransformerMixin):
    """
    Обучает KNN на широте/долготе предсказывать медианный доход.
    Добавляет колонку 'knn_income' в X.
    """

    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def fit(self, X, y=None):
        # === ВАШ КОД ЗДЕСЬ ===
        # Используем только Latitude, Longitude как фичи
        # Цель — 'MedInc' (медианный доход)
        # self.knn.fit(...)

        self.knn.fit(X[["Latitude", "Longitude"]], X["MedInc"])
        return self

    def transform(self, X):
        # === ВАШ КОД ЗДЕСЬ ===
        # 1. Сделать предсказание knn_income
        # 2. Вернуть копию X с добавленной колонкой
        X_out = X.copy()
        # X_out["knn_income"] = ...
        X_out["knn_income"] = self.knn.predict(X_out[["Latitude", "Longitude"]])
        return X_out


# Тестируем трансформер отдельно
knn_trans = KNNIncomeTransformer(n_neighbors=10)
try:
    knn_trans.fit(X_train_small, y_train_small)
    X_train_aug = knn_trans.transform(X_train_small.head())
    print(f"Трансформер сработал. Колонки: {list(X_train_aug.columns)}")
except Exception as e:
    print(f"Ошибка: {e}")

print("TODO: Реализовать трансформер полностью\n")


# ============================================================
# УПРАЖНЕНИЕ 5: Автоматический поиск подготовки через RandomizedSearchCV
#
# Используйте RandomizedSearchCV для поиска лучших параметров
# ПРЕОБРАЗОВАНИЯ ДАННЫХ (не модели):
# - Полиномиальные признаки (degree=2 или 3)
# - Выбор признаков (SelectPercentile или SelectFromModel)
# - Стандартизация (вкл/выкл)
#
# Оборачивайте всё в Pipeline и ищите лучшую комбинацию.
# ============================================================
print("-" * 60)
print("УПРАЖНЕНИЕ 5: Поиск лучшей подготовки данных")
print("-" * 60)

# === ВАШ КОД ЗДЕСЬ ===
# Подсказка: создайте Pipeline с шагами вида:
# [
#   ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#   ('select', SelectPercentile(percentile=50)),
#   ('scale', StandardScaler()),
#   ('model', Ridge())
# ]
# и ищите лучшие параметры через RandomizedSearchCV

print("TODO: Реализовать упражнение 5\n")
preprocessing = Pipeline(
    [
        ("poli", PolynomialFeatures(degree=2, include_bias=False)),
        ("sel", SelectPercentile(percentile=50)),
        ("scl", StandardScaler()),
        ("model", Ridge(random_state=42)),
    ]
)
params = {
    "poli__degree": [2, 3],  # степень полинома
    "sel__percentile": [30, 50, 70],  # сколько % признаков оставить
}

rzscv = RandomizedSearchCV(
    preprocessing, params, n_iter=10, cv=3, n_jobs=4, scoring="neg_mean_squared_error"
)
rzscv.fit(X_train, y_train)
y_pred = rzscv.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"The best params:{rzscv.best_params_}")
print(rmse)
# ============================================================
# УПРАЖНЕНИЕ 6: StandardScalerClone своими руками
#
# Реализуйте класс StandardScalerClone с нуля:
# - fit(): запоминает mean_ и scale_ (std), feature_names_in_ (если X — DataFrame)
# - transform(): (X - mean_) / scale_
# - inverse_transform(): X * scale_ + mean_
# - get_feature_names_out(input_features=None):
#   * если input_features передан — проверить длину,
#     проверить совпадение с feature_names_in_ (если есть),
#     вернуть input_features
#   * если input_features=None — вернуть feature_names_in_
#     (если есть) или np.array(["x0", "x1", ...])
# ============================================================
print("-" * 60)
print("УПРАЖНЕНИЕ 6: StandardScalerClone с нуля")
print("-" * 60)


class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # === ВАШ КОД ЗДЕСЬ ===
        # 1. Проверить, что X — 2D
        # 2. Вычислить mean_ (np.mean по axis=0) и scale_ (np.std по axis=0, ddof=0)
        # 3. Запомнить n_features_in_
        # 4. Если X — DataFrame, сохранить feature_names_in_ (как np.array колонок)
        if len(X.shape) == 2:
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0, ddof=0)
            self.n_features_in_ = X.shape[1]
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = np.array(X.columns)
        else:
            raise ValueError(f"X должен быть 2D, получено {len(X.shape)}D")
        return self

    def transform(self, X):
        # === ВАШ КОД ЗДЕСЬ ===
        # Проверить, что n_features_in_ совпадает
        # Вернуть (X - mean_) / scale_
        if not hasattr(self, "mean_"):
            raise RuntimeError("nigga plz run fit() first") 
        if X.shape[1] != self.n_features_in_:
            raise ValueError()

        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        # === ВАШ КОД ЗДЕСЬ ===
        # Проверить n_features_in_
        # Вернуть X * scale_ + mean_
        if not hasattr(self, "mean_"):
            raise RuntimeError("nigga plz run fit() first") 
        if X.shape[1] != self.n_features_in_:
            raise ValueError()

        return X *self.scale_ + self.mean_

    def get_feature_names_out(self, input_features=None):
        '''
        - get_feature_names_out(input_features=None):
      * если input_features передан — проверить длину,
        проверить совпадение с feature_names_in_ (если есть),
        вернуть input_features
      * если input_features=None — вернуть feature_names_in_
        (если есть) или np.array(["x0", "x1", ...])
        '''
        # === ВАШ КОД ЗДЕСЬ ===

        if input_features is not None:
            if hasattr(self,"feature_names_in"):
                if len(input_features) == len(self.feature_names_in_):
                    return input_features

        else:
            if hasattr(self,'feature_names_in_'):
                return self.feature_names_in_

        return np.array(["x0", "x1"])  # заглушка


# Тест
scaler_clone = StandardScalerClone()
try:
    X_sample = X_train_small[["MedInc", "HouseAge"]].iloc[:100]
    scaler_clone.fit(X_sample)
    X_scaled = scaler_clone.transform(X_sample)
    X_back = scaler_clone.inverse_transform(X_scaled)

    print(f"feature_names_in_: {scaler_clone.feature_names_in_}")
    print(f"mean_: {scaler_clone.mean_}")
    print(f"scale_: {scaler_clone.scale_}")
    print(f"inverse_transform error: {np.abs(X_sample.values - X_back).max():.2e}")

    names = scaler_clone.get_feature_names_out()
    print(f"get_feature_names_out(): {names}")
except Exception as e:
    print(f"Ошибка: {e}")

print("TODO: Реализовать класс полностью\n")
#3*60 + 44 минуты болы потрачено на 6 заданий

# ============================================================
# УПРАЖНЕНИЕ 7: Полный ML-проект — свой датасет
#
# Выберите один из датасетов:
# A) Vehicle — предсказать цену подержанного авто (age, km, make, model...)
# B) Bike Sharing — предсказать кол-во арендованных велосипедов (cnt)
#    по дню недели, времени, погоде
# C) Любой другой регрессионный датасет
#
# Пройдите все этапы из главы 2:
# 1. Быстрый взгляд на данные (info, describe, hist)
# 2. Визуализация (корреляции, scatter matrix)
# 3. Обработка пропусков, создание новых признаков
# 4. Кодирование категорий, масштабирование
# 5. Пайплайн подготовки
# 6. Обучение нескольких моделей (LinearRegression, DecisionTree, RandomForest, SVR)
# 7. Подбор гиперпараметров (GridSearchCV / RandomizedSearchCV)
# 8. Оценка на тесте
# ============================================================
print("-" * 60)
print("УПРАЖНЕНИЕ 7: Полный ML-проект")
print("-" * 60)

# === ВАШ КОД ЗДЕСЬ ===
# Реализуйте полный пайплайн для выбранного датасета
#
# Пример структуры:
#
# # 1. Загрузка
# df = pd.read_csv("bike_sharing.csv")
#
# # 2. Исследование
# print(df.info())
# print(df.describe())
#
# # 3. Разделение
# X = df.drop("cnt", axis=1)
# y = df["cnt"].copy()
# X_train, X_test, y_train, y_test = train_test_split(...)
#
# # 4. Пайплайн
# num_pipeline = Pipeline([...])
# full_pipeline = ColumnTransformer([...])
#
# # 5. Модель
# model = RandomForestRegressor(...)
#
# # 6. Оценка
# predictions = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, predictions))
# print(f"Test RMSE: {rmse:.4f}")

print("TODO: Реализовать упражнение 7\n")


# ============================================================
print("=" * 60)
print("Файл готов! Заполняйте TODOs и запускайте.")
print("=" * 60)
