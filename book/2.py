import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import ml

data = pd.read_csv('./datasets/housing/housing.csv',sep=',')
df= pd.DataFrame(data)