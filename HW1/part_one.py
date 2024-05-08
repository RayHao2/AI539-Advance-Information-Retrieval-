import os
import json
from nltk.tokenize import word_tokenize
import nltk
from string import punctuation
from nltk.stem import PorterStemmer
import re
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.stats import linregress
import pickle

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_fraction(s):
    fraction_pattern = re.compile(r'^\d+/\d+$')
    return bool(fraction_pattern.match(s))

def is_stop_word(token):
    stop_words = set([
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in',
        'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the',
        'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with'
    ])

    # Convert the word to lowercase for case-insensitive comparison
    lowercase_word = token.lower()

    return lowercase_word in stop_words
def remove_non_alphanumeric(input_string):
    # Use regular expression to keep only English letters and numbers
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result

def lienar_regression(word, count):
    # word = np.array(word)
    count = np.array(count)
    word = np.arange(1, len(count) + 1)
    # Perform log transformation
    log_word = np.log(word)
    log_count = np.log(count)

    # Perform linear regression on log-transformed data
    slope, intercept, r_value, p_value, std_err = linregress(log_word, log_count)

    # Plot original data and regression line on log-log scale
    plt.figure(figsize=(8, 6))
    plt.scatter(word, count, color='blue', label='Data')
    plt.plot(word, np.exp(intercept) * word**slope, color='red', label='Linear Regression')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Word')
    plt.ylabel('DF')
    plt.title('Word vs DF')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Slope is ", slope)


def part_one():
    path = "/Users/raymondchen/Desktop/AI539_IR/yelp"
    tokens = {}
    TTF_count = {}
    DF_count = {}
    stemmer = PorterStemmer()
    yelp_store_list = os.listdir(path)
    #Loop through each yelp file
    for yelp_store in yelp_store_list:
        store_path = path + "/" + yelp_store
        with open(store_path, 'r') as file:
            data = json.load(file)
        reviews = data["Reviews"]
        #Loop through each review in the yelp _ file
        for review in reviews:
            doc = review['Content']
            doc_ID = review['ReviewID']
            tokens = word_tokenize(doc) #token the review content
            #Loop through each token and normal it
            for token in tokens:
                token = remove_non_alphanumeric(token)
                if len(token) == 0:
                    continue
                elif is_integer(token):
                    token = "NUM"
                elif is_stop_word(token):
                    continue
                else:
                    token = stemmer.stem(token)
                    token = token.lower()
                TTF_count[token] = 1 + TTF_count.get(token, 0)
                if token in DF_count and DF_count[token] is not None:
                    DF_count[token].add(doc_ID)
                else:
                    DF_count[token] = set()
                    DF_count[token].add(doc_ID)

    with open('TFF_count.pkl', 'wb') as f:
        pickle.dump(TTF_count, f)
    with open('DF_count.pkl', 'wb') as f:
        pickle.dump(DF_count, f)
    # with open("output.txt", "w") as file:
    #     sys.stdout = file
    #     print(TFF_count)
    #     print("==================================================================")
    #     print(DF_count)

    # graph TTF
    sorted_TTF_count = dict(
        sorted(TTF_count.items(), key=lambda item: item[1], reverse=True))
    word = list(sorted_TTF_count.keys())
    TTF = list(sorted_TTF_count.values())
    # plt.figure(figsize=(8, 6))
    # plt.loglog(word, count, marker='o', linestyle='-', color='b')
    # # Set the x-axis to log scale
    # plt.xscale('log')
    # plt.xlabel('X-axis (word)')
    # plt.ylabel('Y-axis (count)')
    # plt.title('TTF graph')
    # plt.show()
    
    # lienar_regression(word,TTF)
    # with open("TFF_count.txt", "w") as file:
    #     for c in count:
    #         file.write(str(c) + "\n")
    
    # graph DF
    for key in DF_count:
        DF_count[key] = len(DF_count[key])
    sorted_DF_count = dict(
        sorted(DF_count.items(), key=lambda item: item[1], reverse=True))
    word = list(sorted_DF_count.keys())
    DF = list(sorted_DF_count.values())

    lienar_regression(word,DF)
    
    # plt.figure(figsize=(8, 6))
    # plt.loglog(word, count, marker='o', linestyle='-', color='b')

    # # Set the x-axis to log scale
    # plt.xscale('log')

    # plt.xlabel('X-axis (word)')
    # plt.ylabel('Y-axis (count)')
    # plt.title('DF graph')
    # plt.show()


def main():
    part_one()



if __name__ == "__main__":
    main()