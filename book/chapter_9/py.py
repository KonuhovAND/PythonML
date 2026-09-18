import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

perc = Perceptron(random_state=42)
iris = load_iris(as_frame=True)
x = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target == 0


perc.fit(x, y)
x_new = [[2, 0.5], [3, 1]]
y_pred = perc.predict(x_new)
print(y_pred)
