import pandas as pd
from nltk.corpus import stopwords
import nltk

data = pd.read_csv('description.csv')
doc = data['description'][2]
# doc = "a fully stacked library is a bookworm s dream a lahore based author awais khan shared a picture of his endless book collection on twitter.awais bookshelf runs from one end of the wall to another and has over books some did not even fit in the picture shared by him on the microblogging website.he shared the picture with the caption i counted my books this week.i now have over books some are not pictured here .guess how many i ve actually read i counted my books this week.i now have over books some are not pictured here .guess how many i ve actually read is the author of no honour and shared the picture of his beautifully stacked books in vertical and horizontal order and asked his followers guess how many of the books he has actually read.the tweet has been liked almost times and retweeted times along with hundreds of comments.in another tweet he revealed how many books he has read.he wrote thanks a lot for your wonderful messages.they mean the world to me.for those of you who are asking i ve read over books.thanks a lot for your wonderful messages.they mean the world to me.for those of you who are asking i ve read over books.people took to the comments section to share their reactions to the beautiful photo.one user wrote the last count of my books was also i dont count now bcoz of the guilt of having an endless tbr.another commented what a delightful sight.buying a book is so addictive.the reading part will eventually happen.great i too recently took a count and have about books in my home library but have ran out of space so storing the new incomings randomly presently reading the new map by daniel yergin and lords of desert by james barr on tussle between britain france us in middle east wrote a third user.twittersome users also wondered how much money was spent on making the neatly stacked collection of books."

from sklearn.feature_extraction.text import CountVectorizer

n_gram_range = (1,1)
stop_words = set(stopwords.words("english"))
new_words = set(['i','tweet', 'with', 'me','read', 'would', 'also', 'k', 'know', 'along', 'lot', 'isnt', 'didnt', 'dont', 'youve', 'im', 'even', 'many', 'my', 'took', 'u', 'he', 'hemyself', '&', "i'm", '-', 'via', 'cc', 'we', 'our', 'ive', 'ihaventours', 'ourselves', 'you', "you've", '–', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but','twitter', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'the', 'to', 'of', 'and', 'a', 'in', 'is', 'that', 'for', 'on', 'was', 'has', 'with', 'as', 'it', 'he', 'from', 'have', 'are', 'be', 'by', 'this', 'his', 'at', 'an', 'i', 'not', 'will', 'who', 'been', 'had', 'we', 'their', 'said', 'her', 'also', 'they', 'you', 'but', 'after', 'were', 'which', 'one', 'she', 'people', 'or', 'all', 'about', 'can', 'its', 'more', 'when', 'new', 'per', 'out', 'over', 'would', 'there', 'like', 'india', 'up', 'so', 'if', 'my', 'first', 'read:', 'no', 'our', 'how', 'even', 'while', 'only', 'some', 'other', 'what', 'two', 'him', 'time', 'get', 'just', 'last', 'many', 'against', 'into', 'being', 'them', 'than', 'now', 'do', 'us', 'may', 'your', 'these', 'could', 'made', 'during', 'most', 'since', 'where', 'any', 'man', 'told', 'take', 'around', 'such', 'due', '-', 'then', 'rs', 'second', 'through', 'under', 'got', 'because', 'make', 'those', 'years', 'every', 'know', 'shared', '&', 'help', 'should', 'still', 'seen', 'going', 'very', 'me', 'media', 'back', 'started', 'work', 'used', 'before', 'case'])

k = stop_words.union(new_words)


# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=k).fit([doc])
candidates = count.get_feature_names()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

from sklearn.metrics.pairwise import cosine_similarity

top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)




