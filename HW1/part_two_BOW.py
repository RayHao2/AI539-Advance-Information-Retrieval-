import multiprocessing
import time
from gensim.models import Word2Vec
import gensim.downloader
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import json
from gensim.test.utils import common_texts
import pickle
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def remove_non_alphanumeric(input_string):
    # Use regular expression to keep only English letters and numbers
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result


def get_vocab():
    current_dir = os.getcwd()
    path = os.path.join(os.path.dirname(current_dir), "yelp")
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    vocabulary = set()
    # Prepare documents
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        # Loop through each review in the yelp _ file
        for review in reviews:
            doc = review['Content']
            tokens = word_tokenize(doc)  # token the review content
            # Loop through each token and normal it
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif token in stop_words:
                    continue
                elif is_integer(token):
                    token = "NUM"
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                vocabulary.add(token)
    vocabulary = sorted(vocabulary)
    with open("pklSave/vocabulary.pkl", "wb") as f:
        pickle.dump(vocabulary, f)
    vocabulary_index = {word: i for i, word in enumerate(vocabulary)}
    with open("pklSave/vocabulary_index.pkl", "wb") as f:
        pickle.dump(vocabulary_index, f)


def process_doc():
    current_dir = os.getcwd()
    path = os.path.join(os.path.dirname(current_dir), "yelp")
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    reviewsContents = {}
    # Prepare documents
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        # Loop through each review in the yelp _ file
        for review in reviews:
            doc = review['Content']
            reviewID = review['ReviewID']
            processed_review = []
            tokens = word_tokenize(doc)  # token the review content
            # Loop through each token and normal it
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif token in stop_words:
                    continue
                elif is_integer(token):
                    token = "NUM"
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                processed_review.append(token)
            reviewsContents[reviewID] = processed_review

    with open("pklSave/reviewsContents.pkl", "wb") as f:
        pickle.dump(reviewsContents, f)


def BOW(vocab_index):
    current_dir = os.getcwd()
    path = os.path.join(os.path.dirname(current_dir), "yelp")
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    BOWs = []
    # Prepare documents
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        # Loop through each review in the yelp _ file
        for review in reviews:
            doc = review['Content']
            reviewID = review['ReviewID']
            BOW = set()
            tokens = word_tokenize(doc)  # token the review content
            # Loop through each token and normal it
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif token in stop_words:
                    continue
                elif is_integer(token):
                    token = "NUM"
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                BOW.add(vocab_index[token])
            BOW_tuple = (reviewID, BOW)
            BOWs.append(BOW_tuple)

    with open("pklSave/BOWs.pkl", "wb") as f:
        pickle.dump(BOWs, f)


def process_query(vocab_index, vocab):
    with open("pklSave/query.pkl", "rb") as f:
        query = pickle.load(f)
    # constrct query index
    query_index = []
    for q in query:
        query_set = set()
        for word in q:
            index = vocab_index[word]
            query_set.add(index)
        query_index.append(query_set)

    for index in query_index:
        for i in index:
            print(vocab[i], end=" ")
        print()

    with open("pklSave/query_index.pkl", "wb") as f:
        pickle.dump(query_index, f)

def look_up():
    pass

def main():
    # get_vocab()
    with open("pklSave/vocabulary.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("pklSave/vocabulary_index.pkl", "rb") as f:
        vocab_index = pickle.load(f)

    # process_doc()
    with open("pklSave/reviewsContents.pkl", "rb") as f:
        reviewsContents = pickle.load(f)
        # for review in reviewsContents:
        #     print(reviewsContents[review])
        #     break

    # BOW(vocab_index)
    with open("pklSave/BOWs.pkl", "rb") as f:
        BOWs = pickle.load(f)
        # for index in BOWs[0][1]:
        #     print(vocab[index], sep=" ")

    # process_query(vocab_index, vocab)
    with open("pklSave/query_index.pkl", "rb") as f:
        query_index = pickle.load(f)
        print(query_index)
        
    look_up(vocab, vocab_index, reviewsContents, BOWs, query_index)


if __name__ == "__main__":
    main()
