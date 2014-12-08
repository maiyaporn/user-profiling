from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import csv

class TextProcessing(object):
	"""docstring for UserProfilling"""
	def removePunctuation(self, text):
		if type(text) is not str:
			text = text.encode('utf-8')
		text_ = text.replace("/", " ")
		text_ = text.replace(".", " ")
		text_ = text_.translate(string.maketrans("",""), string.punctuation)
		return text_

	def removeStopWords(self, text):
		if type(text) is str:
			text = text.decode('utf-8')
		customStop = [i.decode('utf-8') for i in ["you're", "you'll", "wasn't", "shouldn't", "won't", "couldn't", "they're", "don't", "doesn't", "didn't", "hadn't", "you've", "haven't", "weren't", "it's", "can't", "i'm", "we're", "they've", "i've", "we've", "wouldn't", "aren't", "isn't"]]
		stop = stopwords.words('english') + customStop
		text_ = " ".join(i for i in text.split() if i.lower() not in stop)
		for i in text_.split():
			if i in customStop:
				print i
		return text_

	def stemText(self, text):
		#stemmer = PorterStemmer()
		stemmer = SnowballStemmer("english")
		text_ = " ".join(stemmer.stem(i) for i in text.split())
		return text_

	def lemmatizeText(self, text):
		if type(text) is str:
			text = text.decode('utf-8')
			
		lmtzr = WordNetLemmatizer()
		text_ = " ".join(lmtzr.lemmatize(i) for i in text.split())
		return text_

	def preprocess(self, rawtext):
		text = removeStopWords(rawtext)
		text = removePunctuation(text)
		text = lemmatizeText(text)
		return text


def main():
	up = UserProfilling()

if __name__ == '__main__':
	main()		