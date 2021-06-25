import RAKE
import operator
import pandas as pd

stopdir = 'stopwords.txt'
rake_obj = RAKE.Rake(stopdir)
data = pd.read_csv('description.csv')
doc = data['description'][0]

def sort_tuple(tup):
    tup.sort(key = lambda x:x[1],reverse = True)
    return tup
keywords = sort_tuple(rake_obj.run(doc))[0:5]
print(keywords)



