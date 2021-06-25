import json
import string
from urllib.request import urlopen
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk

import mysql.connector

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

df = pd.read_csv("IT contentIDs and URLs - ITIdsUrls.csv")
# print(df[0:2])

def jsonload(url):
        with urlopen(url) as response:
            source = response.read()
        data = json.loads(source)
        soup = BeautifulSoup((data['data']['detail']['description']),"html.parser")
        for data in soup(['blockquote']):
            data.decompose()
        article = (soup.find_all(['p','h3','h1','h2','h6']))
        desc = " "
        for y in article:
            if(len(y.text) > 20):
                # print(y.text.strip())
                desc = desc + y.text.strip()
        # print(desc + "\n")
        # desc = re.sub('http://\S+|https://\S+|pic.twitter.com/[\w]*|@[\w]*|', '', desc)
        desc = re.sub(r'\s+', ' ',desc)
        # desc = desc.encode('ascii', 'ignore').decode("utf-8")


        # desc = desc.replace('-'," ")
        # desc = desc.replace('.'," ")
        # translator = str.maketrans('', '', string.punctuation)
        # desc =  desc.translate(translator)
        return(str(desc))

docs = []
for url in df["API URL"]:
    docs.append(jsonload(url))
ids = list(df['content_id'])
data = pd.DataFrame(list(zip(ids,docs)),columns =['content_id', 'description'])
data.to_csv(r'preprocessed_desc.csv', index=False)

# print(jsonload(df['API URL'][0]))