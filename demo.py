import pandas as pd

root_dir = "../Dataset/Train.csv"

data = pd.read_csv(root_dir)
print(data.iloc[0, -2])
