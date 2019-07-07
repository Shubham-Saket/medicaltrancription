import os
import pickle
import re
import pandas as pd
import numpy as np
from keras_preprocessing import sequence
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize as wp
from gensim.models import Doc2Vec, Word2Vec
import json
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from keras.models import Sequential, Model
from keras import layers
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional

from medical_transcrition_nlp.Constants import WordConstants
from medical_transcrition_nlp.Constants import DocConstants
from medical_transcrition_nlp.Constants.CMConstants import KEYWORDLENGTH, MAXKEYWORD, RFC_RANGEFIT, RFC_NESTIMATOR


class train_classifier_model():
    def __init__(self, file_csv, is_model='rfc'):

        self.dataset = pd.read_csv(file_csv)
        self.stop_words = list(get_stop_words('en'))
        nltk_words = list(stopwords.words('english'))
        self.stop_words.extend(nltk_words)
        self.dataset.dropna(inplace=True)
        self.dataset.reset_index(inplace=True)
        self.dataset = self.dataset.loc[:, ~self.dataset.columns.str.contains('^Unnamed')]

        self.word_model = Word2Vec.load(os.path.join('data', WordConstants.FILENAME))

        self.doc_model = Doc2Vec.load(os.path.join('data', DocConstants.FILENAME))

        self.all_keyword = []
        for doc in self.dataset['keywords']:
            for keywords in doc.split(','):
                if keywords.strip() != '':
                    self.all_keyword.append(keywords.strip())
        self.all_keyword = list(set(self.all_keyword))
        self.is_model = is_model
        self.prepare_data()

    def prepare_data(self):
        total_vec = []
        label = []
        if self.is_model not in ['blstm', 'seqnn']:
            for document in range(len(self.dataset['transcription'])):
                keyword_vec = []
                keyword_counter = 0
                for keyword in self.all_keyword:
                    if (keyword_counter <= MAXKEYWORD):
                        for word in wp(keyword):
                            for j in ["\\", '(', ')', '*', '+', '[', ']']:
                                word = word.replace(j, '\\' + j)
                            cmp = re.compile(word.lower())
                            if cmp.findall(self.dataset['transcription'][
                                               document].lower()) != []:  # and (keyword_counter <= MAXKEYWORD):
                                try:
                                    keyword_vec += list(self.word_model.wv[word][:KEYWORDLENGTH])
                                    keyword_counter += 1
                                except:
                                    continue
                if keyword_counter < MAXKEYWORD:
                    for i in range(MAXKEYWORD - keyword_counter):
                        keyword_vec += list(np.zeros(KEYWORDLENGTH))

                vec = self.doc_model.infer_vector([self.dataset['transcription'][document].lower()])
                # sims = self.doc_model.most_similar([vec], topn=len(self.doc_model.docvecs))

                total_vec.append(list(vec) + keyword_vec)
                label.append(self.dataset['medical_specialty'][document])

            self.train_data = pd.DataFrame(total_vec)
            self.train_data.fillna(value=0.0, inplace=True)

            self.label_data = pd.DataFrame(label, columns=['y_data'])

        elif self.is_model == 'seqnn':
            self.tokenizer = Tokenizer(num_words=300)
            self.tokenizer.fit_on_texts(self.dataset["transcription"])
            pickle.dump(self.tokenizer, open(os.path.join('data', str(self.is_model) + '_tokenizer.pickle'), 'wb+'))
            self.train_data = self.tokenizer.texts_to_matrix(self.dataset["transcription"], mode='tfidf')
            self.train_data = pd.DataFrame(self.train_data)
            self.label_data = pd.get_dummies(self.dataset['medical_specialty'], columns=['medical_specialty'])
            y_labels = sorted(self.label_data.columns)
            y_labels_dic = {}

            for i in range(len(y_labels)):
                y_labels_dic[y_labels[i]] = i
            json.dump(y_labels_dic, open(os.path.join('data', str(self.is_model) + 'label.json'), 'w+'))
        elif self.is_model == 'blstm':
            self.vocab_size, embedding_size = self.word_model.wv.syn0.shape
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(self.dataset["transcription"])
            pickle.dump(self.tokenizer, open(os.path.join('data', str(self.is_model) + '_tokenizer.pickle'), 'wb+'))
            sequences = self.tokenizer.texts_to_sequences(self.dataset["transcription"])

            self.train_data = sequence.pad_sequences(sequences, maxlen=300, padding='post')
            self.new_embedding = pd.np.zeros((self.vocab_size, embedding_size))
            self.new_embedding = self.embedding_matrix_generate(self.new_embedding, self.tokenizer, self.word_model)
            self.label_data = pd.get_dummies(self.dataset['medical_specialty'], columns=['medical_specialty'])
            y_labels = sorted(self.label_data.columns)
            y_labels_dic = {}

            for i in range(len(y_labels)):
                y_labels_dic[y_labels[i]] = i
            json.dump(y_labels_dic, open(os.path.join('data', str(self.is_model) + '_label.json'), 'w+'))

    def embedding_matrix_generate(self, new_embedding, tokenizer, word_model):
        for word, i in tokenizer.word_index.items():
            try:
                embedding_vector = word_model.wv[word]
                if embedding_vector is not None:
                    new_embedding[i] = embedding_vector
            except:
                continue
        return new_embedding

    def train(self):
        print(self.is_model)
        if self.is_model == 'rfc':
            self.classifier = RFC(n_estimators=RFC_NESTIMATOR, warm_start=True)
            self.classifier.fit(self.train_data, self.label_data)
            # for i in range(RFC_RANGEFIT):
            #     self.classifier.n_estimators += 1
            #     self.classifier.fit(self.train_data, self.label_data)
            predict = self.classifier.predict(self.train_data)
            class_report = classification_report(self.label_data, predict)

            print(class_report)
        if self.is_model == 'svc':
            self.classifier = SVC(kernel='linear')
            self.classifier.fit(self.train_data, self.label_data)
            predict = self.classifier.predict(self.train_data)
            class_report = classification_report(self.label_data, predict)
            print(class_report)
        if self.is_model == 'xgb':
            self.classifier = XGBClassifier()
            self.classifier.fit(self.train_data, self.label_data)
            predict = self.classifier.predict(self.train_data)
            class_report = classification_report(self.label_data, predict)
            print(class_report)

        if self.is_model == 'seqnn':
            input_shape = len(self.train_data.columns)
            self.classifier = Sequential()
            # Input - Layer
            self.classifier.add(layers.Dense(100, activation="relu", input_shape=(input_shape,)))
            # Hidden - Layers
            self.classifier.add(layers.Dropout(0.3, noise_shape=None, seed=None))
            self.classifier.add(layers.Dense(100, activation="relu"))
            self.classifier.add(layers.Dropout(0.2, noise_shape=None, seed=None))
            self.classifier.add(layers.Dense(100, activation="relu"))
            # Output- Layer
            self.classifier.add(layers.Dense(len(self.label_data.columns), activation="sigmoid"))
            self.classifier.summary()

            self.classifier.compile(optimizer='adam', loss="binary_crossentropy",
                                    metrics=["accuracy"])
            self.classifier.fit(self.train_data, self.label_data, epochs=50, verbose=1)
            predict = self.classifier.predict(self.train_data)
            class_report = classification_report(self.label_data, predict.round())
            print(class_report)
        if self.is_model == 'blstm':
            inputs = Input(name='inputs', shape=[300])
            layer = Embedding(self.vocab_size, 300, input_length=300,
                              weights=[self.new_embedding], trainable=True)(inputs)
            layer = Bidirectional(LSTM(50))(layer)
            layer = Dense(50, name='FC1')(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(0.2)(layer)
            layer = Dense(len(self.label_data.columns), name='out_layer')(layer)
            layer = Activation('sigmoid')(layer)
            self.classifier = Model(inputs=inputs, outputs=layer)
            self.classifier.compile(loss='categorical_crossentropy', optimizer='adam',
                                    metrics=['accuracy'])

            self.classifier.summary()

            train_loss = self.classifier.fit(self.train_data,
                                             self.label_data,
                                             batch_size=50,
                                             epochs=10,
                                             validation_split=0.2).history["loss"]
            y_predict = self.classifier.predict(self.train_data)
            class_report = classification_report(self.label_data, y_predict.round())
            print(class_report)
        self.save_model()

    def save_model(self):
        if self.is_model in ['seqnn', 'blstm']:
            self.classifier.save(os.path.join('data', str(self.is_model) + '.h5'))
        else:
            pickle.dump(self.classifier, open(os.path.join('data', str(self.is_model) + '.pickle'), 'wb+'))
