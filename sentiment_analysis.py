""" This code does sentiment analysis

Time-stamp: <2016-07-23 01:12:55 yaning>

Author: Yaning Liu
Main used modules are nltk, beautifulsoup, scikit-learn, pandas

"""

import numpy as np
import os.path
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import pandas as pd
from multiprocessing import Pool
import logging
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from itertools import repeat
import time
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
import operator
import sklearn.metrics


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

class sentiment_analysis:

    def __init__(self, NLP_model, ML_method,
                 maxfeature=5000, use_pool=False, pool_size=None):
        """The initializer of the sentiment_analysis class

        :param NLP_model: the natural language processing model, e.g.,
        'BagOfWords', 'Word2Vec', 'BagOfCentroids'
        :param ML_method: the Maching learning methods used, e.g.,
        'RandomForest', 'LogisticRegression', 'MultinomialNB', SGDClassifier
        'SVM'
        :param maxfeature: the maximum number of features
        :param maxfeature: the maximum number of feature to use
        :param nltk_path: the nltk path to append

        """
        self.NLP_model = NLP_model
        self.ML_method = ML_method
        self.maxfeature = maxfeature
        self.use_pool = use_pool
        self.pool_size = pool_size

    def construct_NLP_model(self, review_list, **kwargs):
        """Construct natural language processing model, assume

        :param review_list: the list of reviews, each review is a list of words
        :returns: vectorizer and train_data_features as class
        members
        :rtype: train_data_features: array of size nsamples x nfeatures

        """

        import review_processing as rp

        # Training process of word model
        if self.NLP_model == 'BagOfWords':
            print('construct_NLP_model: Creating bag of words...', flush=True)
            # for Bag of Words, the data should be a list of string
            review_list = [' '.join(rev) for rev in review_list]
            keys_in = kwargs.keys()
            if 'vectorizer_type' in keys_in:
                if kwargs['vectorizer_type'] == 'CountVectorizer':
                    print('construct_NLP_model: Using CountVectorizer')
                    self.vectorizer = CountVectorizer(
                        analyzer='word', tokenizer=None, preprocessor=None,
                        stop_words=None, max_features=self.maxfeature)
                elif kwargs['vectorizer_type'] == 'TfidfVectorizer':
                    print('construct_NLP_model: Using TfidfVectorizer')
                    self.vectorizer = TfidfVectorizer(
                        analyzer='word', tokenizer=None, preprocessor=None,
                        stop_words=None, max_features=self.maxfeature)
            else:
                print('construct_NLP_model: No vectorizer type provided for '
                      'Word2Vec. CountVectorizer is used')
                self.vectorizer = CountVectorizer(
                    analyzer='word', tokenizer=None, preprocessor=None,
                    stop_words=None, max_features=self.maxfeature)

            self.train_data_features = self.vectorizer.fit_transform(
                review_list)
            self.train_data_features = self.train_data_features.toarray()

            # vocab = self.vectorizer.get_feature_names()
            # dist = np.sum(self.train_data_features, axis=0)
            # for tag, count in zip(vocab, dist):
            #     print(count, tag)
        elif self.NLP_model == 'Word2Vec':
            print('construct_NLP_model: using Word2Vec...', flush=True)
            # default values
            self.w2v_num_features = 300    # Word vector dimensionality
            keys_in = kwargs.keys()
            if 'w2v_num_features' in keys_in:
                self.w2v_num_features = kwargs['w2v_num_features']
            else:
                print('construct_NLP_model: No w2v_num_features provided for '
                      'Word2Vec. Default value is used')
            if 'w2v_model_name' in keys_in:
                self.w2v_model = Word2Vec.load(kwargs['w2v_model_name'])

                self.train_data_features = self.get_avg_feature_vecs(
                    review_list, self.w2v_model, self.w2v_num_features,
                    self.use_pool, self.pool_size)
            else:
                sys.exit('construct_NLP_model: No w2v model name provided for '
                         'Word2Vec!')
        elif self.NLP_model == 'BagOfCentroids':
            print('construct_NLP_model: using BagOfCentroids...', flush=True)
            keys_in = kwargs.keys()
            if 'w2v_model_name' in keys_in:
                self.w2v_model = Word2Vec.load(kwargs['w2v_model_name'])
            else:
                sys.exit('construct_NLP_model: No w2v model name provided for '
                         'BagOfCentroids!')

            if 'cluster_ratio' in keys_in:
                cluster_ratio = kwargs['cluster_ratio']
            else:
                cluster_ratio = 5
                print('construct_NLP_model: No cluster ratio provided for '
                      'BagOfCentroids. Default value of 5 is used')

            if 'cluster_method' in keys_in:
                cluster_method = kwargs['cluster_method']
            else:
                cluster_method = 'KMeans'
                print('construct_NLP_model: No cluster method provided for '
                      'BagOfCentroids. Default value of KMeans is used')

            print('Clustering words for BagOfCentroid...')
            sys.stdout.flush()
            t0 = time.time()
            self.word_centroid_map = self.word_clustering(
                self.w2v_model, cluster_ratio, cluster_method, self.pool_size
            )
            print('Clustering words finished. It takes {} seconds'.
                  format(time.time()-t0))
            sys.stdout.flush()

            print('Clustering bags of centroids for BagOfCentroid...')
            sys.stdout.flush()
            t0 = time.time()
            self.train_data_features = self.create_bags_of_centroids(
                review_list, self.word_centroid_map, self.use_pool, self.pool_size)
            print('Clustering bags of centroids finished. It takes {} seconds'.
                  format(time.time()-t0))
            sys.stdout.flush()
        else:
            sys.exit('construct_NLP_model: NLP_model type not supported yet!')

    def NLP_model_predict(self, review_list_test):
        """Given new reviews, compute the features, e.g., bag of words

        :param review_list_test: the list contains the test reviews
        :returns: test_data_features
        :rtype: 2d vectors (arrays)

        """

        print('Predicting test data features with {}...'.format(self.NLP_model),
              flush=True)

        # for Bag of Words, the data should be a list of string
        if self.NLP_model == 'BagOfWords':
            review_list_test = [' '.join(rev) for rev in review_list_test]
            test_data_features = self.vectorizer.transform(review_list_test)
            test_data_features = test_data_features.toarray()
        elif self.NLP_model == 'Word2Vec':
            test_data_features = self.get_avg_feature_vecs(review_list_test,
                                                           self.w2v_model,
                                                           self.w2v_num_features,
                                                           self.use_pool,
                                                           self.pool_size)
        elif self.NLP_model == 'BagOfCentroids':
            test_data_features = self.create_bags_of_centroids(review_list_test,
                                                               self.word_centroid_map,
                                                               self.use_pool,
                                                               self.pool_size)

        return test_data_features

    def train_ML_model(self, sentiment, **kwargs):
        """train a machine learning model

        :param sentiment: list/1d array, the sentiment
        :returns: a ML model
        :rtype: a ML class object

        """
        if self.ML_method == 'RandomForest':
            print('Training the data with Random Forest classifier...', flush=True)
            if 'n_estimators' not in kwargs.keys():
                print('No n_estimators provided for Random Forest. '
                      'By default, 100 will be used!')
                RF_n_est = 100
            else:
                RF_n_est = kwargs['n_estimators']
            self.random_forest_model = RandomForestClassifier(
                n_estimators=RF_n_est, n_jobs=self.pool_size)
            self.random_forest_model = self.random_forest_model.fit(
                self.train_data_features, sentiment)
            self.ML_model = self.random_forest_model
        elif self.ML_method == 'LogisticRegression':
            print('Training the data with Logistic Regression classifier...',
                  flush=True)
            self.logistic_regression_model = LogisticRegression(
                n_jobs=self.pool_size)
            self.logistic_regression_model = self.logistic_regression_model.fit(
                self.train_data_features, sentiment)
            self.ML_model = self.logistic_regression_model
        elif self.ML_method == 'MultinomialNB':
            print('Training the data with Multinomial Naive Bayes classifier...',
                  flush=True)
            self.multinomial_nb_model = MultinomialNB()
            self.multinomial_nb_model = self.multinomial_nb_model.fit(
                self.train_data_features, sentiment)
            self.ML_model = self.multinomial_nb_model
        elif self.ML_method == 'SGDClassifier':
            print('Training the data with stochastic gradient descent '
                  'classifier...', flush=True)
            self.SGD_model = MultinomialNB(n_jobs=self.pool_size)
            self.SGD_model = self.SGD_model.fit(
                self.train_data_features, sentiment)
            self.ML_model = self.SGD_model
        elif self.ML_method == 'SVM':
            print('Training the data with Support Vector Machine classifier...',
                  flush=True)
            self.SVM_model = svm.SVC(probability=True)
            self.SVM_model = self.SVM_model.fit(
                self.train_data_features, sentiment)
            self.ML_model = self.SVM_model

    def predict_ML_model(self, test_data_features):
        """Machine learning predition

        :param dict_list_test: test list of dictionaries
        :returns: predicted sentiment
        :rtype: numpy array (1d)

        """
        import review_processing as rp

        print('Predicting the data with classifier {}...'.format(self.ML_method),
              flush=True)

        nitems = test_data_features.shape[0]

        print('The size of test data is {0}'.format(nitems), flush=True)

        if self.ML_method == 'RandomForest':
            sentiment_pred = self.random_forest_model.predict(test_data_features)
        elif self.ML_method == 'LogisticRegression':
            sentiment_pred = self.logistic_regression_model.predict(test_data_features)
        elif self.ML_method == 'MultinomialNB':
            sentiment_pred = self.multinomial_nb_model.predict(test_data_features)
        elif self.ML_method == 'SGDClassifier':
            sentiment_pred = self.SGD_model.predict(test_data_features)
        elif self.ML_method == 'SVM':
            sentiment_pred = self.SVM_model.predict(test_data_features)
        else:
            sys.exit('predict_ML_model: ML_method not known!')

        return sentiment_pred

    def get_learning_curve(self, review_list, sentiment,
                           train_sizes=np.linspace(.1, 1.0, 5),
                           scoring_type='accuracy', cv_in=None,
                           **kwargs):
        """Machine learning model cross validation. Compute:
        Area under Receiver operating characterstic (ROC) Curve (AUC):
        roc_auc (metrics.roc_auc_score)
        f1 (metrics.f1_score)
        accuracy (metrics.accuracy_score)

        :param review_list: all data, i.e., training and testing, X, in
        a list of words
        :param sentiment: all sentiment, training and testing, in 1d array/list
        :param train_sizes: a 1d array, specify the portions of training
        data, e.g., np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        :param scoring_type: string, specify the scoring type, e.g, 'roc_auc',
        'f1', 'accuracy'
        :param cv_in: integer, the number of folds,
        or a cross_validation object generator, obtained from cross_validation
        :returns: predicted sentiment
        :rtype: numpy array (1d)

        """
        import review_processing as rp

        print('Computing the learning curve with {0} and {1} classifier...'
              .format(self.NLP_model, self.ML_method), flush=True)

        nitems = len(review_list)
        print('The size of data is {0}'.format(nitems), flush=True)

        if isinstance(cv_in, int):
            cv = cross_validation.StratifiedKFold(sentiment, n_folds=cv_in,
                                                  random_state=88)
        if cv_in is None:
            cv = cross_validation.StratifiedKFold(sentiment)

        cv = list(cv)
        max_training_size = len(cv[0][0])
        actual_training_sizes = np.floor(train_sizes * max_training_size)

        score_all_sz = []
        for sz in actual_training_sizes:
            print('Training data size {}:'.format(sz), flush=True)
            mean_score = {}
            if 'roc_auc' in scoring_type:
                mean_score['roc_auc'] = 0
            if 'f1' in scoring_type:
                mean_score['f1'] = 0
            if 'accuracy' in scoring_type:
                mean_score['accuracy'] = 0
            for cv1 in cv:
                review_list_tra = operator.itemgetter(*(cv1[0][:sz]))(review_list)
                review_list_val = operator.itemgetter(*(cv1[1]))(review_list)
                sentiment_tra = operator.itemgetter(*(cv1[0][:sz]))(sentiment)
                sentiment_val = operator.itemgetter(*(cv1[1]))(sentiment)
                print('training and validation sizes are {0} and {1}'
                      .format(len(review_list_tra), len(review_list_val)), flush=True)

                self.construct_NLP_model(review_list_tra, **kwargs)
                test_data_features = self.NLP_model_predict(review_list_val)
                self.train_ML_model(sentiment_tra, **kwargs)

                if 'roc_auc' in scoring_type:
                    probas = self.ML_model.predict_proba(test_data_features)
                    mean_score['roc_auc'] += sklearn.metrics.roc_auc_score(
                        sentiment_val, probas[:, 1])
                if 'f1' in scoring_type:
                    sentiment_pred = self.predict_ML_model(test_data_features)
                    mean_score['f1'] += sklearn.metrics.f1_score(sentiment_val,
                                                                 sentiment_pred)
                if 'accuracy' in scoring_type:
                    sentiment_pred = self.predict_ML_model(test_data_features)
                    sc = sklearn.metrics.accuracy_score(sentiment_val,
                                                        sentiment_pred)
                    mean_score['accuracy'] += sklearn.metrics.accuracy_score(
                        sentiment_val, sentiment_pred)

            for key in mean_score.keys():
                mean_score[key] /= len(cv)

            score_all_sz.append(mean_score)

        return score_all_sz

    @staticmethod
    def get_feature_vec(wordlist, w2v_model, num_features,
                        model_vocab=None):
        """Function to average all of the word vectors in a given review

        :param wordlist: a list of words from a single review
        :param w2v_model: the trained word2vec model
        :param num_features: the number of features in word2vec
        :param model_vocab: set, the vocabulary of w2v model
        :returns: feature_vec, the normalized (averaged) word features
        :rtype: a 1d numpy vector of size num_features

        """
         # Pre-initialize an empty numpy array for speed
        feature_vec = np.zeros((num_features, ), dtype='float32')

        nwords = 0

        # index2word is a list that contains the names of the words
        # in the model's vocabulary. Convert it to a set for speed

        if model_vocab is None:
            index2word_set = set(w2v_model.index2word)
        else:
            index2word_set = model_vocab

        # Loop over each word in the reviewand if it is in the model's
        # vocabulary, add its feature vector to the total
        for word in wordlist:
            if word in index2word_set:
                nwords += 1
                feature_vec = np.add(feature_vec, w2v_model[word])

        # Divide the result by the number of words to get the average
        if nwords != 0:
            feature_vec = np.divide(feature_vec, nwords)

        return feature_vec

    @staticmethod
    def get_avg_feature_vecs(reviews, w2v_model, num_features,
                             use_pool=False, pool_size=1):
        """Given a set of reviews (each one a list of words), calculate

        :param reviews: a list of a list of words
        :param w2v_model: word2vec model
        :param num_features: number of features
        :param use_pool: bolean if use pool
        :param pool_size: the size of the pool
        :returns: review_feature_vecs
        :rtype: a 2d numpy vector

        """

        index2word_set = set(w2v_model.index2word)

        if not use_pool:
            counter = 0
            reviews_feature_vecs = np.zeros((len(reviews), num_features),
                                            dtype="float32")

            for review in reviews:
                reviews_feature_vecs[counter] = sentiment_analysis.get_feature_vec(
                    review, w2v_model, num_features, index2word_set)
                counter += 1
        elif use_pool:
             pool = Pool(pool_size)
             reviews_feature_vecs = pool.starmap(
                 sentiment_analysis.get_feature_vec, zip(
                     reviews, repeat(w2v_model), repeat(num_features),
                     repeat(index2word_set)))
             pool.close()

             reviews_feature_vecs = np.asarray(reviews_feature_vecs)

        return reviews_feature_vecs

    @staticmethod
    def word_clustering(w2v_model, cluster_ratio, cluster_method, pool_size=1):
        """Clustering for word vectors

        :param w2v_model: the w2v model
        :param cluster_ratio: number of clusters = number of words//cluster_ratio
        :param 'cluster_method': the clustering method
        :param pool_size: number of cpu to use
        :returns: word_centroid_map
        :rtype: a dictionary dict(zip( model.index2word, idx ))

        """

        word_vectors = w2v_model.syn0
        num_clusters = word_vectors.shape[0] // cluster_ratio

        if cluster_method == 'KMeans':
            kmeans_clustering = KMeans(n_clusters = num_clusters, n_jobs=pool_size)
            idx = kmeans_clustering.fit_predict(word_vectors)
        else:
            sys.exit('word_clustering: clustering method {0} is not '
                     'supportet yet'.format(cluster_method))

        word_centroid_map = dict(zip(w2v_model.index2word, idx))

        return word_centroid_map

    @staticmethod
    def create_a_bag_of_centroids(wordlist, word_centroid_map):
        """Create a bag of centroids for a single review

        :param wordlist: a list of words transformed from a single review
        :param word_centroid_map: the map or words to centroid index
        :returns: bag_of_centroids for a single review
        :rtype: a 1d numpy array

        """

        num_centroids = max(word_centroid_map.values()) + 1
        bag_of_centroids = np.zeros(num_centroids, dtype='float32')

        for word in wordlist:
            if word in word_centroid_map:
                idx = word_centroid_map[word]
                bag_of_centroids[idx] += 1

        return bag_of_centroids

    @staticmethod
    def create_bags_of_centroids(listwordlist, word_centroid_map,
                                 use_pool=False, pool_size=1):
        """Create bags of centroids for multiple reviews

        :param listwordlist: a list of a list of words transformed from many
        reviews
        :param word_centroid_map: the map or words to centroid index
        :param use_pool: boolean, if use pool
        :param pool_size: the size of the pool
        :returns: bags_of_centroids for many reviews
        :rtype: a 2d numpy array

        """
        num_centroids = max(word_centroid_map.values()) + 1
        num_reviews = len(listwordlist)
        # bag_of_centroids = np.zeros(num_centroids, dtype='float32')

        if not use_pool:
            bags_of_centroids = np.zeros((num_reviews, num_centroids),
                                        dtype='float32')
            counter = 0
            for wordlist in listwordlist:
                bag_tmp = sentiment_analysis.create_a_bag_of_centroids(
                    wordlist, word_centroid_map)
                bags_of_centroids[counter] = bag_tmp
                counter = counter + 1

        elif use_pool:
            pool = Pool(pool_size)
            bags_of_centroids = pool.starmap(
                sentiment_analysis.create_a_bag_of_centroids,
                zip(listwordlist, repeat(word_centroid_map)))
            pool.close()

            bags_of_centroids = np.asarray(bags_of_centroids)

        return bags_of_centroids
