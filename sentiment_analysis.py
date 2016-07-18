""" This code does sentiment analysis

Time-stamp: <2016-07-16 16:52:56 yaning>

Author: Yaning Liu
Main used modules are nltk, beautifulsoup, scikit-learn, pandas

"""

import numpy as np
import os.path
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import pandas as pd
from multiprocessing import Pool

class sentiment_analysis:

    def __init__(self, NLP_model, ML_method, review_col_name,
                 sentiment_col_name, training_file_name=None,
                 maxfeature=5000, use_pool=False, pool_size=None):
        """The initializer of the sentiment_analysis class

        :param NLP_model: the natural language processing model, e.g.,
        'BagOfWords', 'Word2Vec'
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

    def construct_NLP_model(self, df=None, dict_list=None):
        """Construct natural language processing model, assume
        the reviews have been processed

        :param df: the loaded, processed, clean data frame.
        If data have been loaded into dataframe, then pass in df, and
        set training_file_name and dict_list to be None.
        Otherwise, set df and dict_list to be None and pass in
        training_file_name, or only pass in dict_list
        :param dict_list: the list of dictionaries
        :returns: sentiment, vectorizer and train_data_features as class
        members
        :rtype: train_data_features: array of size nsamples x nfeatures

        """
        import review_processing as rp
        # get words
        if df is not None:
            # nitems = df.shape[0]
            col_names = df.columns.values
            if self.review_col_name not in col_names or \
               self.sentiment_col_name not in col_names:
                sys.exit('construct_NL_model: The name {0}/{1} cannot be '
                         'found'.format(self.review_col_name,
                                        self.sentiment_col_name))
            review_list = df[self.review_col_name].values.tolist()
            self.sentiment = df[self.sentiment_col_name].values
        elif dict_list is not None:
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

        # Training process of Bag of Words
        if self.NLP_model == 'BagOfWords':
            print('construct_NLP_model: Creating bag of words...')
            # for Bag of Words, the data should be a list of string
            review_list = [' '.join(rev) for rev in review_list]
            self.vectorizer = CountVectorizer(analyzer='word',
                                              tokenizer=None,
                                              preprocessor=None,
                                              stop_words=None,
                                              max_features=self.maxfeature)
            self.train_data_features = self.vectorizer.fit_transform(
                review_list)
            self.train_data_features = self.train_data_features.toarray()

            # vocab = self.vectorizer.get_feature_names()
            # dist = np.sum(self.train_data_features, axis=0)
            # for tag, count in zip(vocab, dist):
            #     print(count, tag)

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

        return sentiment_pred
