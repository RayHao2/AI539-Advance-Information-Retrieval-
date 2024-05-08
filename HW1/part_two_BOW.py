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

def Process_doc_get_all_words():
    current_dir = os.getcwd()
    path = os.path.join(os.path.dirname(current_dir), "yelp")
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

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
