from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

x_iris, y_iris = load_iris(as_frame=True).data, load_iris(as_frame=True).target
pca_pipline = make_pipeline(StandardScaler(), PCA())
x_iris_rotated = pca_pipline.fit_transform(x_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(x_iris_rotated, y_iris)
