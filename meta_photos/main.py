import pandas as pd

train_data = pd.read_csv('data/train.csv')

id = train_data[train_data.keys()[:1]]
pawpurality = train_data[train_data.keys()[-1]]
metadata = train_data[train_data.keys()[1:-1]]

#TODO finish this version 