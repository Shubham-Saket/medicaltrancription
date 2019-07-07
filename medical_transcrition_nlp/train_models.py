import os
import sys

sys.path.append('../')

from medical_transcrition_nlp.train_word_vec import train_word_vec
from medical_transcrition_nlp.train_doc_vec import train_doc_vec
from medical_transcrition_nlp.train_classifier_model import train_classifier_model

if __name__ == '__main__':
    filecsv = 'mtsamples.csv'

    wvmodel = train_word_vec(filecsv)
    wvmodel.train()
    wvmodel.save_model()

    docmodel = train_doc_vec(filecsv)
    docmodel.train()
    docmodel.save_model()

    classifier_svm = train_classifier_model(filecsv, is_model='svc')
    classifier_svm.train()
    classifier_svm.save_model()

    classifier_rfc = train_classifier_model(filecsv, is_model='rfc')
    classifier_rfc.train()
    classifier_rfc.save_model()

    classifier_seqnn = train_classifier_model(filecsv, is_model='seqnn')
    classifier_seqnn.train()
    classifier_seqnn.save_model()

    classifier_blstm = train_classifier_model(filecsv, is_model='blstm')
    classifier_blstm.train()
    classifier_blstm.save_model()
