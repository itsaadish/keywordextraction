from summarizer import Summarizer
import pandas as pd

doc = pd.read_csv('preprocessed_desc.csv')
body = (doc['description'][1])


model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
print(full)