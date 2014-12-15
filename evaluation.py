from gensim import corpora, models, similarities, matutils
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
import pickle
import logging
import csv
import os
from scipy.stats.stats import pearsonr

class Evaluation(object):
	"""docstring for Evaluation"""
	def __init__(self):
		self.train_corpus = corpora.MmCorpus('tmp/train-corpus.mm')

	def loadLdaModel(self, n):
		model = 'tmp/lda_n' + str(n) + '_testing_corpus.lda'
		return models.LdaModel.load(model)

	def loadFromPickle(self, pickleFile):
		with open(pickleFile, 'rb') as f:
			return pickle.load(f)

	def readCsvToList(self, csvfile):
		with open(csvfile, 'rb') as f:
			reader = csv.reader(f)
			return list(reader)[0]

	def getUserRestaurants(self, uid):
		directory = 'label/' + str(uid) + '_bus.csv'
		return self.readCsvToList(directory)

	# Calculate pearson correlation score
	def getCorrelation(self, data, uid, sid):
		u_res = data[data['user_id'] == uid]
		s_res = data[data['user_id'] == sid]
		common_res = set(u_res.business_id.values).intersection(s_res.business_id.values)
		corr = (0, 0)
		if len(common_res) >= 10:
			x = []
			y = []
			for r in common_res:
				x_ratings = u_res[u_res['business_id'] == r]['stars_rev']
				if len(x_ratings) > 1:
					x.append(np.average(x_ratings))
				else:
					x.append(int(x_ratings))

				y_ratings = s_res[s_res['business_id'] == r]['stars_rev']
				if len(y_ratings) > 1:
					y.append(np.average(y_ratings))
				else:
					y.append(int(y_ratings))
			corr = pearsonr(x, y)
			#print corr
		return [len(u_res), len(s_res), len(common_res), corr[0], corr[1]]

	def getSimilarities(self, n, pathToIndex):
		# Load testing corpus
		test_corpus = corpora.MmCorpus('tmp/test-corpus.mm')
		doc_idx_map = self.loadFromPickle('tmp/doc-idx-map')
		data = DataFrame(read_json('data50.json'))

		if os.path.isfile(pathToIndex):
			index = similarities.Similarity.load(pathToIndex)
		else:
			model = self.loadLdaModel(n)
			# Transform corpus to lda space and Build an index it - the order of document should be preserved
			index = similarities.Similarity(pathToIndex, model[test_corpus], num_features=n)
			#similarities.MatrixSimilarity(model[test_corpus], num_features=n)
			index.save(pathToIndex)

		simFile = open('tmp/sim_' + str(n) + '.csv', 'wb')
		wr1 = csv.writer(simFile)

		index.num_best = 6
		for docno, sims in enumerate(index):
			uid = doc_idx_map.get(docno)
			for sdocno, score in sims[1:]:
				sid = doc_idx_map.get(sdocno)
				if uid == sid:
					continue
				corr = self.getCorrelation(data, uid, sid)
				if corr[3] > 0.7:
					topicFile = open('tmp/sim_topic_' + str(n) + '_' + str(docno) + '_' + str(sdocno) + '.csv', 'wb')
					wr2 = csv.writer(topicFile)
					# Get topic ditributions
					wr2.writerow([uid, sid, score] + corr)
					wr2.writerow([uid, index.vector_by_id(docno)])
					wr2.writerow([sid, index.vector_by_id(sdocno)])
				wr1.writerow([uid, sid, score] + corr)

	def hellinger(self, vec1, vec2):
		return np.sqrt(0.5 * ((vec1 - vec2)**2).sum())

	def getSimilaritiesByHellinger(self, n):
		# Load testing corpus
		test_corpus = corpora.MmCorpus('tmp/test-corpus.mm')
		doc_idx_map = self.loadFromPickle('tmp/doc-idx-map')
		data = DataFrame(read_json('data50.json'))

		simFile = open('tmp/sim_hl_' + str(n) + '.csv', 'wb')
		wr1 = csv.writer(simFile)

		model = self.loadLdaModel(n)
		precomputed_vec2s = np.sqrt(matutils.corpus2dense(model[test_corpus], num_terms=n, num_docs=len(test_corpus)).transpose())
		for vid, vdoc in enumerate(precomputed_vec2s):
			sims = [(tid, self.hellinger(vdoc, tdoc)) for tid, tdoc in enumerate(precomputed_vec2s)]
			sims = sorted(sims, key=lambda item: item[1])
			uid = doc_idx_map.get(vid)
			for sdocno, score in sims[1:6]:
				sid = doc_idx_map.get(sdocno)
				corr = self.getCorrelation(data, uid, sid)
				wr1.writerow([uid, sid, score] + corr)

	def getTopics(self,n):
		model = self.loadLdaModel(n)
		# Save topics distribution in a file
		topics_tuple = model.show_topics(n, num_words=100, formatted=False)
		with open( 'tmp/' + str(n) + '_topics', 'wb') as f2:
			pickle.dump(topics_tuple, f2)

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	e = Evaluation()

	for n in [60,120,220,340]:
		#e.getTopics(n)
		#e.getSimilarities(n, 'tmp/index/index_n' + str(n))
		e.getSimilaritiesByHellinger(n)

if __name__ == '__main__':
	main()