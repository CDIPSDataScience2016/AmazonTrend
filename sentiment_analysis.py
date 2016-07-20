""" This code does sentiment analysis

Time-stamp: <2016-07-19 16:24:50 yaning>

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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

class sentiment_analysis:

    def __init__(self, NLP_model, ML_method, review_col_name,
                 sentiment_col_name, training_file_name=None,
                 maxfeature=5000, use_pool=False, pool_size=None):
        """The initializer of the sentiment_analysis class

        :param NLP_model: the natural language processing model, e.g.,
        'BagOfWords', 'Word2Vec', 'BagOfCentroids'
        :param ML_method: the Maching learning methods used, e.g.,
        'RandomForest', 'LogisticRegression', 'MultinomialNB', SGDClassifier
        'SVM'
        :param review_col_name: string, the column name of review texts
        :param sentiment_col_name: string, the column name of sentiment values
        maxfeature: the maximum number of features
        :param training_file_name: the file with training data string
        :param maxfeature: the maximum number of feature to use
        :param nltk_path: the nltk path to append

        """
        self.tra_file_name = training_file_name
        self.NLP_model = NLP_model
        self.ML_method = ML_method
        self.review_col_name = review_col_name
        self.sentiment_col_name = sentiment_col_name
        self.maxfeature = maxfeature
        self.use_pool = use_pool
        self.pool_size = pool_size

    def construct_NLP_model(self, dict_list, **kwargs):
        """Construct natural language processing model, assume

        :param dict_list: the list of dictionaries
        :returns: sentiment, vectorizer and train_data_features as class
        members
        :rtype: train_data_features: array of size nsamples x nfeatures

        """

        import review_processing as rp
        # get words
        if dict_list is not None:
            nitems = len(dict_list)
            col_names = list(dict_list[0].keys())
            if self.review_col_name not in col_names or \
               self.sentiment_col_name not in col_names:
                sys.exit('construct_NL_model: The name {0}/{1} cannot be '
                         'found'.format(self.review_col_name,
                                        self.sentiment_col_name))
            review_list = [dic[self.review_col_name]
                           for dic in dict_list]
            self.sentiment = [dic[self.sentiment_col_name] for dic in dict_list]
        elif 'df' in kwargs.keys():
            df = kwargs['df']
            # nitems = df.shape[0]
            col_names = df.columns.values
            if self.review_col_name not in col_names or \
               self.sentiment_col_name not in col_names:
                sys.exit('construct_NL_model: The name {0}/{1} cannot be '
                         'found'.format(self.review_col_name,
                                        self.sentiment_col_name))
            review_list = df[self.review_col_name].values.tolist()
            self.sentiment = df[self.sentiment_col_name].values
        else:
            if self.training_file_name is None:
                sys.exit('construct_NLP_model: traning file name does not '
                         'exist')
            else:
                suffix = os.path.splitext(self.training_file_name)[1][1:]
                if suffix == 'csv':
                    df = pd.read_csv(self.training_file_name)
                    if self.review_col_name not in col_names or \
                       self.sentiment_col_name not in col_names:
                        sys.exit('construct_NL_model: The name {0}/{1} cannot '
                                 ' be found'.format(self.review_col_name,
                                                    self.sentiment_col_name))
                    # nitems = df.shape[0]
                    review_list = df[self.review_col_name].values.tolist()
                    self.sentiment = df[self.sentiment_col_name].values
                elif suffix == 'json':
                    data_dict_list = rp.load_json_data(self.training_file_name)
                    if self.review_col_name not in data_dict_list.keys():
                        sys.exit('construct_NL_model: The name {0} cannot be '
                                 'found'.format(self.review_col_name))
                    review_list = [dic[self.review_col_name] for dic in
                                   data_dict_list]
                    self.sentiment = [dic[self.sentiment_col_name] for dic
                                      in dict_list]
                else:
                    sys.exit('construct_NLP_model: file type not supported '
                             'yet!')

        # Training process of word model
        if self.NLP_model == 'BagOfWords':
            print('construct_NLP_model: Creating bag of words...')
            # for Bag of Words, the data should be a list of string
            review_list = [' '.join(rev) for rev in review_list]
            keys_in = kwargs.keys()
            if 'vectorizer_type' in keys_in:
                if kwargs['vectorizer_type'] == 'CountVectorizer':
                    self.vectorizer = CountVectorizer(
                        analyzer='word', tokenizer=None, preprocessor=None,
                        stop_words=None, max_features=self.maxfeature)
                elif kwargs['vectorizer_type'] == 'TfidfVectorizer':
                    self.vectorizer = TfidVectorizer(
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
                review_list, word_centroid_map, self.use_pool, self.pool_size)
            print('Clustering bags of centroids finished. It takes {} seconds'.
                  format(time.time()-t0))
            sys.stdout.flush()
        else:
            sys.exit('construct_NLP_model: NLP_model type not supported yet!')

    def train_ML_model(self, **kwargs):
        """train a machine learning model

        :returns: a ML model
        :rtype: a ML class object

        """
        if self.ML_method == 'RandomForest':
            print('Training the data with Random Forest classifier...')
            if 'n_estimators' not in kwargs.keys():
                print('No n_estimators provided for Random Forest. '
                      'By default, 100 will be used!')
                RF_n_est = 100
            else:
                RF_n_est = kwargs['n_estimators']
            self.random_forest_model = RandomForestClassifier(
                n_estimators=RF_n_est, n_jobs=self.pool_size)
            self.random_forest_model = self.random_forest_model.fit(
                self.train_data_features, self.sentiment)
        elif self.ML_method == 'LogisticRegression':
            print('Training the data with Logistic Regression classifier...')
            self.logistic_regression_model = LogisticRegression(
                n_jobs=self.pool_size)
            self.logistic_regression_model = self.logistic_regression_model.fit(
                self.train_data_features, self.sentiment)
        elif self.ML_method == 'MultinomialNB':
            print('Training the data with Multinomial Naive Bayes classifier...')
            self.multinomial_nb_model = MultinomialNB()
            self.multinomial_nb_model = self.multinomial_nb_model.fit(
                self.train_data_features, self.sentiment)
        elif self.ML_method == 'SGDClassifier':
            print('Training the data with stochastic gradient descent '
                  'classifier...')
            self.SGD_model = MultinomialNB(n_jobs=self.pool_size)
            self.SGD_model = self.SGD_model.fit(
                self.train_data_features, self.sentiment)
        elif self.ML_method == 'SVM':
            print('Training the data with Support Vector Machine classifier...')
            self.SVM_model = svm.SVC()
            self.SVM_model = self.SVM_model.fit(
                self.train_data_features, self.sentiment)

    def predict_ML_model(self, df_test=None, test_file_name=None,
                         dict_list=None):
        """Machine learning predition

        :param df_test: test data frame, if test file name
        and dict_list= None (not provided)
        :param test_file_name: test file name if test data frame or dict_list
        is None
        :param dict_list_test: test list of dictionaries
        :returns: predicted sentiment
        :rtype: numpy array (1d)

        """
        import review_processing as rp

        print('Predicting the data with Random Forest classifier...')

        if df_test is not None:
            nitems = df_test.shape[0]
            col_names = df_test.columns.values
            if self.review_col_name not in self.col_names_test:
                sys.exit('predict_ML_model: The name {0} cannot be found'.
                         format(self.review_col_name))
            review_list = df_test[self.review_col_name].values.tolist()
        elif test_file_name is not None:
            suffix = os.path.splitext(test_file_name)[1][1:]
            if suffix == 'csv':
                df_test = pd.read_csv(self.test_file_name)
                if self.review_col_name not in col_names:
                    sys.exit('predict_ML_model: The name {0} cannot '
                             ' be found'.format(self.review_col_name))
                nitems = df_test.shape[0]
                review_list = df_test[self.review_col_name].values.tolist()
            elif suffix == 'json':
                data_dict_list = rp.load_json_data(test_file_name)
                nitems = len(data_dict_list)
                if self.review_col_name not in data_dict_list[0].keys():
                    sys.exit('predict_ML_model: The name {0} cannot be '
                             'found'.format(self.review_col_name))
                review_list = [dic[self.review_col_name] for dic in
                               data_dict_list]
        elif dict_list is not None:
            nitems = len(dict_list)
            if self.review_col_name not in dict_list[0].keys():
                sys.exit('predict_ML_model: The name {0} cannot be '
                         'found'.format(self.review_col_name))
            review_list = [dic[self.review_col_name] for dic in
                               dict_list]

        print('The size of test data is {0}'.format(nitems))

        # for Bag of Words, the data should be a list of string
        if self.NLP_model == 'BagOfWords':
            review_list = [' '.join(rev) for rev in review_list]
            test_data_features = self.vectorizer.transform(review_list)
            test_data_features = test_data_features.toarray()
        elif self.NLP_model == 'Word2Vec':
            test_data_features = self.get_avg_feature_vecs(review_list,
                                                           self.w2v_model,
                                                           self.w2v_num_features,
                                                           self.use_pool,
                                                           self.pool_size)
        elif self.NLP_model == 'BagOfCentroids':
            test_data_features = self.get_avg_feature_vecs(review_list,
                                                           self.word_centroid_map,
                                                           self.use_pool,
                                                           self.pool_size)

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
                bag_of_centroids[index] += 1

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

        return bag_of_centroids
