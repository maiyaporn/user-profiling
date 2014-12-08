import logging
from gensim import corpora, models, similarities
import numpy as np
from pandas import *
import matplotlib.pyplot as plt

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# load id->word mapping (the dictionary), one of the results of step 2 above
	#dictionary = corpora.Dictionary.load('tmp/train-corpus.dict')
	dictionary = corpora.Dictionary.load('tmp/test-corpus.dict')

	# load corpus iterator
	#train_corpus = corpora.MmCorpus('tmp/train-corpus.mm')
	#val_corpus = corpora.MmCorpus('tmp/val-corpus.mm')
	test_corpus = corpora.MmCorpus('tmp/test-corpus.mm')

	#grid = dict()
	num_topics = [60,120]
	#num_of_words = sum(cnt for doc in val_corpus for _, cnt in doc)
	for n in num_topics:
		#grid[n] = list()
		#lda = models.ldamodel.LdaModel(train_corpus, id2word=dictionary, num_topics=n, update_every=0, passes=20)
		lda = models.ldamodel.LdaModel(test_corpus, id2word=dictionary, num_topics=n, update_every=0, passes=20)
		
		# model perplexity
		'''
		perplex = lda.log_perplexity(val_corpus)
		print "Perplexity: %s" % perplex
		grid[n].append(perplex)

		perplex_bound = lda.bound(val_corpus)
		per_word_perplex = np.exp2(-perplex_bound/num_of_words)
		print "Per-word Perplexity: %s" % per_word_perplex
		grid[n].append(per_word_perplex)
		'''
		#lda.save('tmp/' + 'lda_n' + str(n) + '_training_corpus.lda')
		lda.save('tmp/' + 'lda_n' + str(n) + '_testing_corpus.lda')

	'''
	df = DataFrame(grid)
	df.to_csv('perplexity.csv')
	print df
	plt.figure(figsize=(14,8), dpi=120)
	plt.subplot(211)
	plt.plot(df.columns.values, df.iloc[0].values, '#007A99')
	plt.xticks(df.columns.values)
	plt.ylabel('Perplexity')
	plt.grid(True)

	plt.subplot(212)
	plt.plot(df.columns.values, df.iloc[1].values, 'b')
	plt.xticks(df.columns.values)
	plt.ylabel('Perplexity')
	plt.xlabel("Number of topics", fontsize='large')
	plt.grid(True)

	plt.savefig('tmp/lda_topic_perplexity.png', bbox_inches='tight', pad_inches=0.1)
	'''

	
if __name__ == '__main__':
	main()
