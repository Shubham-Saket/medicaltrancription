import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


from medical_transcrition_nlp.train_word_vec import train_word_vec
from medical_transcrition_nlp.Constants.DocConstants import MAXEPOCHS,VECSIZE,ALPHA,MINCOUNT,MINALPHA,FILENAME

class train_doc_vec(train_word_vec):

    def applytocol(self):
        pass

    def train(self):
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in
                       enumerate(self.dataset['transcription'])]

        self.model = Doc2Vec(vector_size=VECSIZE,
                        alpha=ALPHA,
                        min_alpha=MINALPHA,
                        min_count=MINCOUNT,
                        dm=1,epochs= MAXEPOCHS)

        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data,
                    total_examples=self.model.corpus_count,epochs=self.model.epochs)

    def save_model(self):
        if not os.path.exists('data'):
            os.mkdir('data')
        self.model.save(os.path.join('data',FILENAME))

    def load_model(self,filename):
        self.pretrained_model = Doc2Vec.load(os.path.join('data',filename))
