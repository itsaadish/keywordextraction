from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import pandas as pd
import re


LANGUAGE = 'english'
stemmer = Stemmer(LANGUAGE)
data = pd.read_csv('preprocessed_desc.csv')

article = data['description'][1]
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

parser = PlaintextParser.from_string(clean(article), Tokenizer(LANGUAGE))
summarizer = TextRankSummarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

num_summary_sentence = 3

for sentence in summarizer(parser.document, num_summary_sentence):
    print (str(sentence))