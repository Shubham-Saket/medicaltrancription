import os
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize as wp
from nltk.stem import WordNetLemmatizer as lemma
from sklearn.feature_extraction.text import TfidfVectorizer


class GenerateKeywords():
    def __init__(self, file_csv):
        self.dataset = pd.read_csv(file_csv)
        self.stop_words = list(get_stop_words('en'))
        nltk_words = list(stopwords.words('english'))
        self.stop_words.extend(nltk_words)
        self.dataset.dropna(inplace=True)
        self.dataset = self.dataset.loc[:, ~self.dataset.columns.str.contains('^Unnamed')]
        self.applytocol()
        self.tfidf_vector()

    def listtostring(self, lst):
        out_string = ''
        for i in lst:
            out_string += ' ' + i
        return out_string

    def applytocol(self):
        self.dataset['description'] = self.dataset['description'].apply(
            lambda x: [item for item in wp(x) if item not in self.stop_words]).apply(
            lambda x: [lemma().lemmatize(item) for item in x]).apply(lambda x: self.listtostring(x))
        self.dataset['transcription'] = self.dataset['transcription'].apply(
            lambda x: [item for item in wp(x) if item not in self.stop_words]).apply(
            lambda x: [lemma().lemmatize(item) for item in x]).apply(lambda x: self.listtostring(x))

    def tfidf_vector(self):
        self.vect = TfidfVectorizer(max_features=300)
        self.vect.fit(self.dataset['transcription'])


    def get_keywords(self):
        fname = self.vect.get_feature_names()
        idf = self.vect.idf_
        for i,j in zip(fname,idf):
            print(i,j)

obj = GenerateKeywords('mtsamples.csv')
