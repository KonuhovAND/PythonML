import numpy as np
import pandas as pd 
import matplotlib.pyplot as mt
import sklearn as sk


rng = np.random.default_rng(seed=42)
m= 200
x = 2 * rng.random((m,1))
y = 4 + 3 * x + rng.standard_normal((m,1))