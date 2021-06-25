from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.utils import get_stop_words
import pandas as pd
import re


LANGUAGE = 'english'
stemmer = Stemmer(LANGUAGE)
data = pd.read_csv('preprocessed_desc.csv')


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
def summary(article):
    parser = PlaintextParser.from_string(article, Tokenizer(LANGUAGE))
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    num_summary_sentence = 3
    doc = ""
    for sentence in summarizer(parser.document, num_summary_sentence):
        doc += str(sentence)
    return(doc)
summaries = []      
for text in data['description']:
    summaries.append(summary(clean(text)))
ids = list(data['content_id'])
data = pd.DataFrame(list(zip(ids,summaries)),columns =['content_id', 'summary'])
data.to_csv(r'summarygenerated.csv', index=False)





    







