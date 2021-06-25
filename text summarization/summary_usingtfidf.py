from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize
import pandas as pd
import re
import numpy as np

data = pd.read_csv('preprocessed_desc.csv')

article = data['description'][0]
def clean(text):
    #Remove punctuations
    # text = re.sub(r'[^a-zA-Z. ]+', ' ', text)
    
    #Convert to lowercase
    # text = text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    # text=re.sub("\\d+"," ",text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('. ','.')
    text = text.replace('.','. ')
    
    
    ##Convert to list from string
    return(text)
sentences = tokenize.sent_tokenize(clean(article))
tfidfVectorizer = TfidfVectorizer()
words_tfidf = tfidfVectorizer.fit_transform(sentences)

# print(words_tfidf)


num_summary_sentence = 3

# Sort the sentences in descending order by the sum of TF-IDF values
sent_sum = words_tfidf.sum(axis=1)
important_sent = np.argsort(sent_sum, axis=0)[::-1]

# Print three most important sentences in the order they appear in the article
for i in range(0, len(sentences)):
    if i in important_sent[:num_summary_sentence]:
        print (sentences[i])

# print(clean(article))

# print(sent_sum)