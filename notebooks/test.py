import dill
import pandas as pd

data = pd.read_csv("../data/train_processed_data.csv")
with open("pipeline.dll", 'rb') as f:
    pipeline =  dill.load(f)

pipeline.run(data)