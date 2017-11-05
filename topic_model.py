from gensim.models import LdaModel
import pickle
import spacy
from random import randint

class topicMod(object):
    
    def __init__(self):
        open_file = open("./pickled_files/test_documents_stripped.pickle", "rb")
        self.test_documents = pickle.load(open_file)
        open_file.close()
        self.test_documents=[text for text in self.test_documents if len(text.split())>150]
        
        open_file = open("./pickled_files/bigram_model.pickle", "rb")
        self.bigram_model = pickle.load(open_file)
        open_file.close()
        
        open_file = open("./pickled_files/trigram_model.pickle", "rb")
        self.trigram_model = pickle.load(open_file)
        open_file.close()
        
        self.nlp = spacy.load('en')
        
        self.lda = LdaModel.load('./pickled_files/lda_final_model')
        
        open_file = open("./pickled_files/trigram_dictionary.pickle", "rb")
        self.trigram_dictionary = pickle.load(open_file)
        open_file.close()     
        
        self.topic_names={
        0: 'miscellaneous',
        1: 'medicine',
        2: 'miscellaneous 1',
        3: 'religion',
        4: 'sports',
        5: 'software',
        6: 'encryption',
        7: 'guns',
        8: 'politics',
        9: 'computing',
        10: 'foreign politics', 
        11: 'politics',
        12: 'cars',
        13: 'politics 1',
        14: 'middle east/cars mix',
        15: 'computing',
        16: 'christianity',
        17: 'space',
        18: 'computing 1',
        19: 'religious/atheist discussion'
        }
               
    def get_document(self):
        randn = randint(0,len(self.test_documents))
        return self.test_documents[randn]
        
    def takeSecond(self,elem):
        return elem[1]

    def punct_space(self,token):
        return token.is_punct or token.is_space or token.is_stop

    def lda_description(self,text, min_topic_freq=0.05):
        parsed_doc = self.nlp(text)

        unigram_doc=[]

        for token in parsed_doc:
            if not self.punct_space(token):
                if token.lemma_ == '-PRON-':
                    unigram_doc.append(token.orth_)
                elif token.lemma_ not in spacy.en.STOP_WORDS:
                    unigram_doc.append(token.lemma_)

        bigram_doc = self.bigram_model[unigram_doc]
        trigram_doc = self.trigram_model[bigram_doc]
        trigram_doc = [term for term in trigram_doc]

        trigram_doc = [term for term in trigram_doc if len(term)>1]

        doc_bow = self.trigram_dictionary.doc2bow(trigram_doc)

        doc_lda = self.lda[doc_bow]

        doc_lda = sorted(doc_lda, key=self.takeSecond, reverse=True)
        
        print("{} \n".format(text))
        print("Topics found: \n")
        
        for topic_number, freq in doc_lda:
            if freq < min_topic_freq:
                break

            # print the most highly related topic names and frequencies
            print('{:35} {}'.format(self.topic_names[topic_number],
                                    round(freq, 3)))