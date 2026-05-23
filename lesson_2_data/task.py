import pandas as pd
from pprint import pprint

data = pd.read_csv("./data.csv",sep=",")
df = pd.DataFrame(data)

uniq= df.nunique().to_dict()
pprint(uniq)
print("*" * 30)
count = df.count()
print(count)
print("*" * 30)
print(df)
print("*" * 30)
catagorio = df[df["Make"].str.len() == 0]
print(catagorio)