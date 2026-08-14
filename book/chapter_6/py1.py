import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris, make_moons
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

x, y = make_moons(n_samples=500, noise=0.3, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", SVC(random_state=42)),
    ]
)
voting_clf.fit(x_train, y_train)


for name, clf in voting_clf.named_estimators_.items():
    print(name, f"={clf.score(x_test, y_test)}")

# >>>lr =0.864
# >>>rf =0.896
# >>>svc =0.896

for name, clf in voting_clf.named_estimators_.items():
    print(
        f"{name} = clf.predict(x_test[:1]) is {clf.predict(x_test[:1])} - {y_test[:1]}"
    )
# >>>lr = clf.predict(x_test[:1]) is [1] - [1]
# >>>rf = clf.predict(x_test[:1]) is [1] - [1]
# >>>svc = clf.predict(x_test[:1]) is [0] - [1]


print(voting_clf.score(x_test, y_test))
# >>>0.912


voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", CalibratedClassifierCV(SVC(random_state=42), ensemble=False)),
    ]
)
voting_clf.voting = "soft"
voting_clf.fit(x_train, y_train)
voting_clf.score(x_test, y_test)
# >>>0.912


bagging_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    random_state=42,
    n_estimators=500,
    max_samples=100,
    n_jobs=-1,
)

bagging_clf.fit(x_train, y_train)
print(bagging_clf.score(x_test, y_test))
# >>> 0.904
print("=*&" * 30)

bagging_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    random_state=42,
    oob_score=True,
    n_estimators=500,
    max_samples=100,
    n_jobs=-1,
)

bagging_clf.fit(x_train, y_train)
print(bagging_clf.oob_score_)  # >>> 0.9253333333333333

print(bagging_clf.score(x_test, y_test))  # >>>0.904
print(accuracy_score(y_test, bagging_clf.predict(x_test)))  # >>>0.904

rnd_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs=-1,
    random_state=42,
)

rnd_clf.fit(x_train, y_train)


iris = load_iris(as_frame=True)
rnc_clf = RandomForestClassifier(random_state=42, n_estimators=500)
rnc_clf.fit(iris.data, iris.target)
for score, name in zip(rnc_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)


ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=30,
    learning_rate=0.5,
    random_state=42,
)

x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=42
)

ada_clf.fit(x_train, y_train)
ada_clf_pred = ada_clf.predict(x_test)
print(root_mean_squared_error(y_test, ada_clf_pred))
print(accuracy_score(y_test, ada_clf_pred))


m = 100
rng = np.random.default_rng(seed=42)
X = rng.random((m, 1)) - 0.5
noise = 0.05 * rng.standard_normal(m)
y = 3 * X[:, 0] ** 2 + noise


tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)

X_new = np.array(
    [
        [-0.4],
        [0.0],
        [0.5],
    ]
)

print(sum(tree.predict(X_new) for tree in [tree_reg1, tree_reg2, tree_reg3]))


gbr = GradientBoostingRegressor(
    max_depth=2, n_estimators=3, learning_rate=1, random_state=42
)
gbr.fit(X, y)
print(gbr.predict(X_new).sum())
print(gbr.score(X, y))
plt.scatter(X, y, color="blue", label="No Label")
plt.savefig("./fig.jpg")


stc_clf = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", CalibratedClassifierCV(SVC(random_state=42), ensemble=False)),
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5,
)
scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)
stc_clf.fit(x_train, y_train)
print(stc_clf.score(x_test, y_test))
