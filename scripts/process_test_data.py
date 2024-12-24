from ..pipelines.pipeline import pipeline
import pandas as pd

test_data = pd.read_csv("../data/test_data.csv")

test_processed_data = pipeline(test_data)
test_processed_data.to_csv("../data/test_processed_data.csv",index=False)