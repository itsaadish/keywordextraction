import json
import string
from urllib.request import urlopen
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

df = pd.read_csv("IT contentIDs and URLs - ITIdsUrls.csv")
# print(df[0:2])

def jsonload(arr):
    desc_list = []
    for x in arr:
        with urlopen(x) as response:
            source = response.read()
        data = json.loads(source)
        soup = BeautifulSoup((data['data']['detail']['description']),"html.parser")
        article = (soup.find_all('p'))
        desc = " "
        
        for y in article:
            # print(y.text)
            desc = desc + y.text.lower()
        # print(desc + "\n")
        desc = re.sub('http://\S+|https://\S+|pic.twitter.com/[\w]*|@[\w]*|[0-9]+', '', desc)
        desc = desc.replace('-'," ")
        desc = desc.replace('.'," ")
        translator = str.maketrans('', '', string.punctuation)
        desc =  desc.translate(translator)
        desc_list.append(desc)
    return(desc_list)
        # print(data['data']['detail']['description'])
# jsonload(list(df["API URL"]))
stopwords = ['i','with', 'me', 'my','u','he',"he" 'myself','',"&","i'm","-","via","cc",'we', 'our','ive','ihavent' 'ours', 'ourselves', 'you',"you've",'–', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def tokenize(descriptions):
    tokenize_list = []
    for desc in descriptions:
        words = desc.split()
        stripped = [w.strip(' ') for w in words]
        clean = [w for w in stripped if w not in stopwords]
        stems = [lemmatizer.lemmatize(word) for word in clean]

        tokenize_list.append(stems)
    return((tokenize_list))

# tokenize(jsonload(list(df["API URL"])))

def frequency(tokenize_list):
    frequencies = []
    for desc in tokenize_list:
        d = dict()
        for word in desc:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
        l = Counter(d).most_common(5)
        frequencies.append([k for k,v in l])
    print(frequencies)

frequency(tokenize(jsonload(list(df["API URL"]))))

        
        




















