from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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


ada_clf = AdaBoostClassifier(random_state=42)
