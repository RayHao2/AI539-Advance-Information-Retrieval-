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


def process_documents():
    current_dir = os.getcwd()
    path = os.path.join(os.path.dirname(current_dir), "yelp")
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    documents = [] # store a list of documents with processed words
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
            normaled_token = []
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
                normaled_token.append(token)
            documents.append(normaled_token)
    with open("pklSave/documents.pkl", "wb") as f:
        pickle.dump(documents, f)


def process_query():
    init_query = [
        "general chicken", "fried chicken", "BBQ sandwiches", "mashed potatoes", "Grilled Shrimp Salad",
        "lamb Shank", "Pepperoni pizza", "brussel sprout salad", "FRIENDLY STAFF", "Grilled Cheese"
    ]
    stemmer = PorterStemmer()
    procssed_query = []
    for query in init_query:
        normal_query = []
        query = word_tokenize(query)
        for token in query:
            token = stemmer.stem(token)
            token = token.lower()
            normal_query.append(token)
        procssed_query.append(normal_query)
    with open("pklSave/query.pkl", "wb") as f:
        pickle.dump(procssed_query, f)


def average_vector(doc, model):
    doc_vector = model.wv[doc]
    return np.mean(doc_vector, axis=0)


def cos_sim(query_vec, doc_vec):
    return np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))


def tfidf_vectorize(model, dct, doc):
    doc_bow = dct.doc2bow(doc)
    vector = model[doc_bow]
    dense_vector = np.zeros(len(dct))
    for idx, value in vector:
        dense_vector[idx] = value
    return dense_vector


def tfidf():
    with open("pklSave/documents.pkl", "rb") as f:
        documents = pickle.load(f)
    with open("pklSave/query.pkl", "rb") as f:
        querys = pickle.load(f)

    train_doc = documents + querys
    dct = Dictionary(train_doc)
    train_doc_corpus = [dct.doc2bow(doc) for doc in train_doc]

    # run first time
    start_time = time.time()
    model = TfidfModel(train_doc_corpus)
    end_time = time.time()
    print("Train time: ", end_time - start_time)
    model.save("tfidf.model")

    with open("output.txt", "w") as f:
        f.write("TFIDF Vector cosine-sim")

    for i in range(len(querys)):
        query_vec = tfidf_vectorize(model, dct, querys[i])
        top_similarities = []
        included_documents = set()
        for j in range(len(documents)):
            if j not in included_documents:
                doc_vec = tfidf_vectorize(model, dct, documents[j])

                sim = cos_sim(query_vec, doc_vec)
                if len(top_similarities) < 3:
                    heapq.heappush(top_similarities, (sim, j))
                    included_documents.add(j)
                else:
                    if sim > top_similarities[0][0]:
                        heapq.heappop(top_similarities)
                        heapq.heappush(top_similarities, (sim, j))
                        included_documents.add(j)
        with open("output.txt", "a") as f:
            print(f"Query: {querys[i]}", file=f)
        for sim in top_similarities:
            with open("output.txt", "a") as f:
                print(f"{sim[0]} {documents[sim[1]]} index: {sim[1]}", file=f)
                
                
def main():
    process_documents()
    process_query()
    tfidf()
    
if __name__ == "__main__":
    main()
    