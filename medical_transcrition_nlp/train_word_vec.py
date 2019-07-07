import os
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize as wp
from nltk.stem import WordNetLemmatizer as lemma

from medical_transcrition_nlp.Constants.WordConstants import MINCOUNT,SIZE,WINDOW,FILENAME

class train_word_vec():
    def __init__(self, file_csv):
        self.dataset = pd.read_csv(file_csv)
        self.stop_words = list(get_stop_words('en'))
        nltk_words = list(stopwords.words('english'))
        self.stop_words.extend(nltk_words)
        self.dataset.dropna(inplace=True)
        self.dataset = self.dataset.loc[:, ~self.dataset.columns.str.contains('^Unnamed')]
        self.applytocol()

    def applytocol(self):
        self.dataset['description'] = self.dataset['description'].apply(
            lambda x: [item for item in wp(x) if item not in self.stop_words]).apply(
            lambda x: [lemma().lemmatize(item) for item in x])
        self.dataset['transcription'] = self.dataset['transcription'].apply(
            lambda x: [item for item in wp(x) if item not in self.stop_words]).apply(
            lambda x: [lemma().lemmatize(item) for item in x])

    def train(self):
        self.model = Word2Vec(self.dataset['transcription'],size=SIZE, window=WINDOW, min_count=MINCOUNT)

    def save_model(self):
        if not os.path.exists('data'):
            os.mkdir('data')
        self.model.save(os.path.join('data',FILENAME))

    def load_model(self,filename):
        self.pretrained_model = Word2Vec.load(os.path.join('data',filename))