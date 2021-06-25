from nltk.corpus.reader.wordnet import Lemma
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from orderedset import OrderedSet
import math
import mysql.connector

lemmatizer = WordNetLemmatizer()

stopdir = ['i', 'with', 'me', 'would', 'also', 'k', 'know', 'along', 'lot', 'isnt', 'didnt', 'dont', 'youve', 'im', 'even', 'many', 'my', 'took', 'u', 'he', 'hemyself', '', '&', "i'm", '-', 'via', 'cc', 'we', 'our', 'ive', 'ihaventours', 'ourselves', 'you', "you've", '–', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but','twitter', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'the', 'to', 'of', 'and', 'a', 'in', 'is', 'that', 'for', 'on', 'was', 'has', 'with', 'as', 'it', 'he', 'from', 'have', 'are', 'be', 'by', 'this', 'his', 'at', 'an', 'i', 'not', 'will', 'who', 'been', 'had', 'we', 'their', 'said', 'her', 'also', 'they', 'you', 'but', 'after', 'were', 'which', 'one', 'she', 'people', 'or', 'all', 'about', 'can', 'its', 'more', 'when', 'new', 'per', 'out', 'over', 'would', 'there', 'like', 'india', 'up', 'so', 'if', 'my', 'first', 'read:', 'no', 'our', 'how', 'even', 'while', 'only', 'some', 'other', 'what', 'two', 'him', 'time', 'get', 'just', 'last', 'many', 'against', 'into', 'being', 'them', 'than', 'now', 'do', 'us', 'may', 'your', 'these', 'could', 'made', 'during', 'most', 'since', 'where', 'any', 'man', 'told', 'take', 'around', 'such', 'due', '-', 'then', 'rs', 'second', 'through', 'under', 'got', 'because', 'make', 'those', 'years', 'every', 'know', 'shared', '&', 'help', 'should', 'still', 'seen', 'going', 'very', 'me', 'media', 'back', 'started', 'work', 'used', 'before', 'case']
dataset = pd.read_csv('description.csv')
doc = dataset['description'][0]


def clean(text):
    #Remove punctuations
    text = re.sub(r'[^a-zA-Z. ]+', ' ', text)
    
    #Convert to lowercase
    text = text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("\\d+"," ",text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('. ','.')
    
    
    ##Convert to list from string
    return(text)

# print(clean(doc))

def get_sentences(text):
    sentence = []
    words = []
    sentence_list = []
    temp = text.strip().split('.')
    temp.remove('')
    for sent in temp:
        words = sent.strip().split(" ") # getting the words in sentences.
        words = [lemmatizer.lemmatize(i,pos = 'v') for i in words if len(i) > 1 if i not in stopdir]
        if(len(words) > 1): 
            sentence.append(words) # sentence is a list of lists. (contains a list of sentences in which each sentence is a list of words)
        sentence_list.append(sent)	
    return sentence, sentence_list


def vectorize(sentence):

	# set of unique words in the whole document.
	unique_words = OrderedSet() 

	for sent in sentence:
		for word in sent:
			
			unique_words.add(word)

	unique_words = list(unique_words) # converting the set to a list to make it easier to work with it. 

	#print(unique_words, len(unique_words))

	# a list of lists that contains the vectorized form of each sentence in the document. 
	vector = list()



	for sent in sentence: # iterate for every sentence in the document
		temp_vector = [0] * len(unique_words) # create a temporary vector to calculate the occurence of each word in that sentence. 
		
		for word in sent: # iterate for every word in the sentence. 

			temp_vector[unique_words.index(word)] += 1	

		vector.append(temp_vector) # add the temporary vector to the list of vectors for each sentence (list of lists)

	#print(vector)	


	return vector, unique_words	

# print(vectorize(sentence))


# function to calculate the tf scores
def tf(vector, sentence, unique_words):

	tf = list()

	no_of_unique_words = len(unique_words) 

	for i in range(len(sentence)):

		tflist = list()
		sent = sentence[i]
		count = vector[i]

		for word in sent:
			'''
			if(count[sent.index(word)] == 0):
				count[sent.index(word)] = 1
			'''
			score = count[sent.index(word)]/ float(len(sent)) # tf = no. of occurence of a word/ total no. of words in the sentence. 

			if(score == 0):
				score = 1/ float(len(sentence))

			tflist.append(score)  

		tf.append(tflist)

	# print(tf)	
	
	return tf	



def idf(vector, sentence, unique_words):

	# idf = log(no. of sentences / no. of sentences in which the word appears).

	no_of_sentences = len(sentence)

	idf = list()

	for sent in sentence:
		
		idflist = list()

		for word in sent:

			count = 0 

			for k in sentence:
				if(word in k):
					count += 1
		

			score = math.log(no_of_sentences/float(count)) 

			idflist.append(score)

		idf.append(idflist)	

	# print(idf)	

	return idf


def tf_idf(tf, idf):

	# tf-idf = tf(w) * idf(w)

	tfidf = [[0 for j in range(len(tf[i]))] for i in range(len(tf))]

	for i in range(len(tf)):
		for j in range(len(tf[i])):

			tfidf[i][j] = tf[i][j] * float(idf[i][j])

	# print(tfidf)		

	return tfidf	


def extract_keywords(tfidf, processed_text):
	
	mapping = {}

	for i in range(len(tfidf)):
		for j in range(len(tfidf[i])):

			mapping[processed_text[i][j]] = tfidf[i][j]

	#print(mapping)

	word_scores = sorted(mapping.values(), reverse = True)
	words = []

	scores_to_word = {}

	for i in range(len(tfidf)):
		for j in range(len(tfidf[i])):

			scores_to_word[tfidf[i][j]] = processed_text[i][j]

	for i in range(len(word_scores)):
		if(word_scores[i] != 0):
			words.append(scores_to_word[word_scores[i]])
		else:
			words.append(scores_to_word[word_scores[i]])
			break

	# # print(words)	

	words = list(OrderedSet(words))

	for i in mapping:
		if(mapping[i] == 0):
			words.append(i)		
	
	return words, mapping


t_clean = clean(doc)	

processed_text, sentence_list = get_sentences(t_clean)

sentence_to_index = {i:k for k, i in enumerate(sentence_list)}
	
vector, unique_words = vectorize(processed_text)

tf = tf(vector, processed_text, unique_words)

idf = idf(vector, processed_text, unique_words)

tfidf = tf_idf(tf, idf)	

keywords, mapping = extract_keywords(tfidf, processed_text)
print(keywords)


# mydb = mysql.connector.connect(
#   host="localhost",
#   user="aadish",
#   password="Sharma@8441",
#   database="",
#   auth_plugin='mysql_native_password'
# )




