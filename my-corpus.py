from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models, similarities
import numpy as np
import glob
import os
import codecs
import string
import pickle

class MyCorpus(object):
	"""docstring for MyCorpus"""
	def __init__(self):
		self.customwords = [i.encode('utf-8') for i in ["n't", "'ve", "'m", "'ll", "'re"]]
		self.stoplists = stopwords.words('english') + self.customwords
		self.lmtzr = WordNetLemmatizer()

	def isPunctuation(self, text):
		if type(text) is not str:
			text = text.encode('utf-8')
		text_ = text.translate(string.maketrans("",""), string.punctuation)
		return bool(len(text_)==0)

	def tokenizeDoc(self, doc):
		tokens = []
		for text in codecs.open(doc, "r", "utf-8"):
			tokens += [self.lmtzr.lemmatize(word) for word in word_tokenize(text.lower()) if len(word) > 3 and word not in self.stoplists and not self.isPunctuation(word)]
		return tokens

	def buildDictionary(self, directory, dictName):
		dictionary = corpora.Dictionary()
		for doc in glob.glob(directory + "/*"):
			dictionary.add_documents([self.tokenizeDoc(doc)])
		dictionary.filter_extremes(no_above=0.7)
		dictionary.compactify()
		dictionary.save(dictName)
		print (dictionary)
		return dictionary

	def buildCorpus(self, directory, dictName, corpusName):
		if os.path.isfile(dictName):
			dictionary = corpora.Dictionary.load(dictName)
		else:
			dictionary = self.buildDictionary(directory, dictName)

		corpus = []
		doc_idx_map = dict()
		n = 0
		for doc in glob.glob(directory + "/*"):
			doc_idx_map[n] = doc.split("/")[1]
			corpus.append(dictionary.doc2bow(self.tokenizeDoc(doc)))
			n += 1
		corpora.MmCorpus.serialize(corpusName, corpus)
		with open('tmp/doc-idx-map', 'wb') as f:
			pickle.dump(doc_idx_map, f)

		print len(corpus)

def main():
	corpus = MyCorpus()
	#corpus.buildCorpus('train', 'tmp/train-corpus.dict', 'tmp/train-corpus.mm')
	#corpus.buildCorpus('val', 'tmp/train-corpus.dict', 'tmp/val-corpus.mm')

	# Build corpus for testing documents - combine train and val
	# Save the order of file read to corpus to make a map of index - user
	corpus.buildCorpus('documents', 'tmp/test-corpus.dict', 'tmp/test-corpus.mm')

if __name__ == '__main__':
	main()