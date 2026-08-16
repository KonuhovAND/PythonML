from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
x, y = iris.data[["petal length (cm)", "petal width (cm)"]].values, iris.target


dtr_clf = DecisionTreeClassifier(random_state=42, max_depth=2)
dtr_clf.fit(x, y)

# export_graphviz(
#     dtr_clf,
#     out_file='my.dot',
#     feature_names=['petal length (cm)','petal width (cm)'],
#     class_names = iris.target_names,
#     rounded=True,
#     filled=True
# )

# Source.from_file('my.dot').render('my.dot', view=True)

print(dtr_clf.predict_proba([[5, 1.5]]).round(3))
