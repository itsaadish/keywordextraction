import json
import string
from urllib.request import urlopen
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import pandas as pd

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
        article = (soup.find_all('p'))
        desc = " "
        for y in article:
            # print(y.text)
            desc = desc + y.text.lower()
        # print(desc + "\n")
        desc = re.sub('http://\S+|https://\S+|pic.twitter.com/[\w]*|@[\w]*|[0-9]+', '', desc)
        desc = desc.replace('-'," ")
        desc = desc.replace('.'," ")
        desc = re.sub(r'\s+', ' ',desc)
        desc = desc.encode('ascii', 'ignore').decode("utf-8")
        translator = str.maketrans('', '', string.punctuation)
        desc =  desc.translate(translator)
        return(desc)
        # print(data['data']['detail']['description'])
# jsonload((df["API URL"][0]))
stopwords = ['i','with','me','would','also','k','know', 'along','lot','isnt','didnt','dont','youve','im','even','many','my','took','u','he',"he" 'myself','',"&","i'm","-","via","cc",'we', 'our','ive','ihavent' 'ours', 'ourselves', 'you',"you've",'–', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",'the', 'to', 'of', 'and', 'a', 'in', 'is', 'that', 'for', 'on', 'was', 'has', 'with', 'as', 'it', 'he', 'from', 'have', 'are', 'be', 'by', 'this', 'his', 'at', 'an', 'i', 'not', 'will', 'who', 'been', 'had', 'we', 'their', 'said', 'her', 'also', 'they', 'you', 'but', 'after', 'were', 'which', 'one', 'she', 'people', 'or', 'all', 'about', 'can', 'its', 'more', 'when', 'new', 'per', 'out', 'over', 'would', 'there', 'like', 'india', 'up', 'so', 'if', 'my', 'first', 'read:', 'no', 'our', 'how', 'even', 'while', 'only', 'some', 'other', 'what',  'two', 'him', 'time', 'get', 'just', 'last', 'many', 'against', 'into', 'being', 'them', 'than', 'now', 'do', 'us', 'may', 'your', 'these', 'could', 'made', 'during', 'most', 'since', 'where', 'any', 'man', 'told', 'take', 'around', 'such', 'due', '-', 'then', 'rs', 'second', 'through', 'under','got', 'because', 'make', 'those', 'years', 'every', 'know', 'shared', '&', 'help', 'should', 'still', 'seen', 'going', 'very', 'me', 'media', 'back', 'started', 'work', 'used', 'before', 'case']
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def tokenize(description):
    tokenize_list = []
    words = description.split()
    stripped = [w.strip(' ') for w in words]
    clean = [w for w in stripped if w not in stopwords]
    stems = [lemmatizer.lemmatize(word) for word in clean]
    cleaned = [w for w in stems if w not in stopwords]
    return((cleaned))

# tokenize(jsonload((df["API URL"][1])))

def frequency(tokenized):
    d = dict()
    for word in tokenized:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
    l = Counter(d).most_common(5)
    return([k for k,v in l])


# keywords = pd.DataFrame(columns =['content_id', 'id','keyword'])


# freq_list = []

# for index,row in df.iterrows():
#     freq = frequency(tokenize(jsonload(row['API URL'])))
#     for i in range(1,len(freq)+1):
#         keywords = keywords.append({'content_id':row['content_id'],'id':i,'keyword':freq[i-1]},ignore_index=True)

# keywords.to_csv(r'keywords.csv', index=False)











mydb = mysql.connector.connect(
  host="localhost",
  user="aadish",
  password="Sharma@8441",
  database = 'mydb',
  auth_plugin='mysql_native_password'
)

mycursor = mydb.cursor()

sql = ("INSERT INTO KEYWORDS(CONTENT_ID, ID, KEYWORD)"
   "VALUES (%s,%s,%s)")

def insertion(sql,data):
    try:
   # Executing the SQL command
        mycursor.execute(sql,data)

   # Commit your changes in the database
        mydb.commit()
        print('inserted')

    except:
   # Rolling back in case of error
        mydb.rollback()

# Closing the connection




for index,row in df.iterrows():# for index,row in df.iterrows():
    freq = frequency(tokenize(jsonload(row['API URL'])))
    for i in range(1,len(freq)+1):
        data = (row['content_id'],i,freq[i-1])
        insertion(sql,data)
    

mydb.close()



















