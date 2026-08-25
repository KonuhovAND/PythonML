# 7. Train and fine-tune a decision tree for the moons dataset by following these
# steps:
# a. Use make_moons(n_samples=10000, noise=0.4) to generate a moons dataset.
# b. Use train_test_split() to split the dataset into a training set and a test set.
# c. Use grid search with cross-validation (with the help of the GridSearchCV
# class) to find good hyperparameter values for a DecisionTreeClassifier.
# Hint: try various values for max_leaf_nodes.
# d. Train it on the full training set using these hyperparameters, and measure
# your model’s performance on the test set. You should get roughly 85% to 87%
# accuracy.
import numpy as np
from scipy.stats import mode
from sklearn.datasets import make_moons
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=1000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)
#
# param_grid = {
#     "max_depth": [3, 5, 7, 10],
#     "max_leaf_nodes": [5, 10, 12, 15, 20, 30, None],
#     "min_samples_split": [2, 3, 4, 6],
#     "max_features": [1, 2, 4, 6, 8],
# }
params_dicision = {
    "max_depth": 3,
    "max_features": 1,
    "max_leaf_nodes": None,
    "min_samples_split": 2,
}
# grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid)
# grid.fit(X_train, y_train)
# print(grid.best_score_, grid.best_params_)
#

# 8. Grow a forest by following these steps:
# a. Continuing the previous exercise, generate 1,000 subsets of the training set,
# each containing 100 instances selected randomly. Hint: you can use ScikitLearn’s ShuffleSplit class for this.
# b. Train one decision tree on each subset, using the best hyperparameter values
# found in the previous exercise. Evaluate these 1,000 decision trees on the test
# set. Since they were trained on smaller sets, these decision trees will likely
# perform worse than the first decision tree, achieving only about 80% accuracy.
# Exercises | 193
# c. Now comes the magic. For each test set instance, generate the predictions of
# the 1,000 decision trees, and keep only the most frequent prediction (you can
# use SciPy’s mode() function for this). This approach gives you majority-vote
# predictions over the test set.
# d. Evaluate these predictions on the test set: you should obtain a slightly higher
# accuracy than your first model (about 0.5 to 1.5% higher). Congratulations,
# you have trained a random forest classifier!
rs = ShuffleSplit(1000, train_size=100, random_state=42)

subsets = []
for subset_id, _ in rs.split(X_train):
    X_subset = X_train[subset_id]
    y_subset = y_train[subset_id]
    subsets.append((X_subset, y_subset))
results = []
predictions = np.empty((len(subsets), len(X_test)), dtype=y_train.dtype)
for train_id, (X_subset, y_subset) in enumerate(subsets):
    dtc = DecisionTreeClassifier(**params_dicision)
    dtc.fit(X_subset, y_subset)
    results.append(dtc.score(X_test, y_test))
    predictions[train_id] = dtc.predict(X_test)

final_pred = mode(predictions, axis=0, keepdims=False).mode
forest_acuracy = np.mean(final_pred == y_test)

print(f"{max(results)} - max result ")
print(f"{min(results)} - min result ")
print(f"{np.mean(results)} - mean result")
print(f"{forest_acuracy}")
