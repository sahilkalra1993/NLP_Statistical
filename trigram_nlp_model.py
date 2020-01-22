# -*- coding: utf-8 -*-

# code courtesy of https://nlpforhackers.io/language-models/
# https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/

# Author: Sahil Kalra

import nltk
nltk.download('reuters')
nltk.download('punkt')


# The Reuters Corpus contains 10,788 news documents totaling 1.3 million words. 
# The documents have been classified into 90 topics, and grouped into two sets, 
# called "training" and "test"; thus, the text with fileid 'test/14826' is a 
# document drawn from the test set. This split is for training and testing 
# algorithms that automatically detect the topic of a document, as we will see 
# in chap-data-intensive.

from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

# Create a placeholder for model
# if the key not in dictionary, return a defaultdictionary which can return
# 0 if queried for a non exiting item
model = defaultdict(lambda: defaultdict(lambda: 0))

# Example:
# model['Sahil'] - defaultdict(<function __main__.<lambda>.<locals>.<lambda>>, {'Kalra': 0})
# model['Sahil']['Kalra'] - 0

"""We first split our text into trigrams with the help of NLTK and then calculate the frequency 
in which each combination of the trigrams occurs in the dataset.
We then use it to calculate probabilities of a word, given the previous two words. 
That’s essentially what gives us our Language Model!
"""

# Count frequency of co-occurance using trigrams
for sentence in reuters.sents():
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1

# Transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

# dict(model[("the", "price")]) <-- To see all the available predictions

# Custom predict function to show the prediction of next word using max probability.
def predict(model, list_words):
    max_prob = 0
    max_key = 0
    for k,v in dict(model[list_words]).items():
        if v>max_prob:
            max_prob=v
            max_key=k
        
    print ("Prediction of next word: ", max_key, max_prob)

predict(model, ("the", "price"))



