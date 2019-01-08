''' 
Building a Simple Chatbot from Scratch in Python (using NLTK)

A chatbot is an artificial intelligence-powered piece of software in a device . 
There are broadly two variants of chatbots: Rule-Based and Self learning.
Self learning is further two types: Retrieval Based or Generative.

'''
import nltk
import numpy as np
import random
import string # to process standard python strings

f = open('Chatbot.txt','r',errors = 'ignore') # corpus 
raw = f.read()
raw = raw.lower()# converts to lowercase
'''
This tokenizer divides a text into a list of sentences
by using an unsupervised algorithm to build a model for abbreviation
words, collocations, and words that start sentences.
'''
nltk.download('punkt') # first-time use only
'''
You can use WordNet alongside the NLTK module to find the meanings of words, synonyms, antonyms, and more.
'''
nltk.download('wordnet') # first-time use only

'''
A tokenizer that divides a string into substrings by splitting on the specified string from punkt.
'''
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer() # PortStemmer
#WordNet is a semantically-oriented dictionary of English included in NLTK.

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    # Term Frequency-Inverse Document Frequency
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') # bag of words model
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input("YOU: ")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")

'''
Thanks and Regards 
Source : https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e

'''