from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    root_mean_squared_error,
)

mnist = load_digits(as_frame=False)
X_data, y_target = mnist.data, mnist.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X_data, y_target, test_size=0.2, random_state=42
)

y_train_7 = y_train == 7

y_test_7 = y_test == 7

sdg_clf = SGDClassifier(random_state=42)
sdg_clf.fit(X_train, y_train_7)
y_pr = sdg_clf.predict(X_test)
# precision_score is 1.0
print(f"precision_score is {precision_score(y_test_7, y_pr)}")
print(f"recall is {recall_score(y_test_7, y_pr)}")  # recall is 0.970
# accuracy_score is 0.99722
print(f"accuracy_score is {accuracy_score(y_test_7, y_pr)}")
print(
    f"root_mean_squared_error is {root_mean_squared_error(y_test_7, y_pr)}"
)  # root_mean_squared_error is 0.0527


sdg_clf = SGDClassifier(random_state=42)

print(
    f"Cross_val_score is {
        cross_val_score(sdg_clf, X_train, y_train_7, cv=3, scoring='accuracy')
    }"
)
# >>> [0.99373695 0.98538622 0.99164927]


dummy = DummyClassifier(random_state=42)
dummy.fit(X_train, y_train_7)
print(any(dummy.predict(X_train)))
# >>> False

print(
    f"Cross_val_score is {
        cross_val_score(dummy, X_train, y_train_7, cv=3, scoring='accuracy')
    }"
)
# >>> [0.89979123 0.89979123 0.89770355]


skfolds = StratifiedKFold(n_splits=3)
for train_index, test_index in skfolds.split(X_train, y_train_7):
    clone_clf = clone(sdg_clf)

    X_train_folds = X_train[train_index]
    y_train_7_folds = y_train_7[train_index]

    X_test_folds = X_train[test_index]
    y_test_7_folds = y_train_7[test_index]

    clone_clf.fit(X_train_folds, y_train_7_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_7_folds)
    print(n_correct / len(y_pred))

# >>>0.9937369519832986
# >>>0.9853862212943633
# >>>0.9916492693110647


y_prediction = cross_val_predict(sdg_clf, X_train, y_train_7, cv=3)
print(
    f"Confusion matrix of prediction is:\n {confusion_matrix(y_train_7, y_prediction)}"
)
# >>> [[1284    8]
# >>> [   6  139]]


y_prediction = cross_val_predict(
    sdg_clf, X_train, y_train_7, cv=3, method="decision_function"
)
fpr, tpr, thresholds = roc_curve(y_train_7, y_prediction)
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], "k:", label="Random clasifier' Roc curve")

plt.savefig("book/chapter_3/ROC.png")
