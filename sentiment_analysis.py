""" This code does sentiment analysis

Time-stamp: <2016-07-13 22:44:28 yaningliu>

Author: Yaning Liu
Main used modules are nltk, beautifulsoup, scikit-learn, pandas

"""

import numpy as np
import os.path
import os.sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


class sentiment_analysis:

    def __init__(self, training_file_name=None,
                 NLP_model, ML_method, review_col_name, sentiment_col_name,
                 maxfeature=5000):
        """The initializer of the sentiment_analysis class

        :param training_file_name: the file with training data string
        :param NLP_model: the natural language processing model, e.g.,
        'BagOfWords', 'Word2Vec'
        :param ML_method: the Maching learning methods used, e.g.,
        'RandomForest'
        :param review_col_name: string, the column name of review texts
        :param sentiment_col_name: string, the column name of sentiment values
        maxfeature: the maximum number of features

        """
        self.tra_file_name = training_file_name
        self.NLP_model = NLP_model
        self.ML_method = ML_method
        self.review_col_name = review_col_name
        self.sentiment_col_name = sentiment_col_name
        self.maxfeature = maxfeature

    def construct_NLP_model(self, df=None):
        """Construct natural language processing model

        :param df: the loaded, processed, clean data frame.
        If data have been loaded into dataframe, then pass in df, and
        set training_file_name to be None. Otherwise, set df to be None and
        pass in training_file_name

        :returns: sentiment, vectorizer and train_data_features as class
        members
        :rtype: train_data_features: array of size nsamples x nfeatures
        """
        import review_processing as rp
        # get words
        if df is not None:
            nitems = df.shape[0]
            col_names = df.columns.values
            if self.review_col_name not in col_names or \
               self.sentiment_col_name not in col_names:
                sys.exit('construct_NL_model: The name {0}/{1} cannot be found'.
                         format(self.review_col_name, self.sentiment_col_name))
            review_list = df[self.review_col_name].values.tolist()
            meaningful_words = map(self.review_to_meaningful_words,
                                   review_list)
            # Get training sentiment values
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
                       self.sentiment_col_name not in col_names::
                        sys.exit('construct_NL_model: The name {0}/{1} cannot '
                                 ' be found'.format(self.review_col_name,
                                                    self.sentiment_col_name))
                    nitems = df.shape[0]
                    review_list = df[review_col_name].values.tolist()
                    meaningful_words = map(self.review_to_meaningful_words,
                                           review_list)
                elif suffix == 'json':
                    data_dict_list = rp.load_data(self.training_file_name)
                    if self.review_col_name not in data_dict_list.keys():
                        sys.exit('construct_NL_model: The name {0} cannot be '
                                 'found'.format(review_col_name))
                    review_list = map(lambda x: x[review_col_name],
                                      data_dict_list)
                    meaningful_words = map(self.review_to_meaningful_words,
                                           review_list)
                else:
                    sys.exit('construct_NLP_model: file type not supported '
                             'yet!')

        # Training process of Bag of Worlds
        if self.NLP_model == 'BagofWords':
            print('construct_NLP_model: Creating bag of words...')
            self.vectorizer = CountVectorizer(analyzer='word',
                                              tokenizer=None,
                                              preprocessor=None,
                                              stop_words=None,
                                              max_features=self.maxfeature)
            self.train_data_features = vectorizer.fit_transform(
                meaningful_words)
            self.train_data_features = train_data_features.toarray()

            # vocab = vectorizer.get_feature_names()
            # dist = np.sum(train_data_features, axis=0)
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
            self.random_forest_model = RandomForestClassifier(n_estimators=100)
            self.random_forest_model = random_forest_model.fit(
                train_data_features, self.sentiment)

    def predict_ML_model(self, df_test=None, test_file_name=None):
        """Machine learning predition

        :param df_val: test data frame, if test file name = None (not provided)
        :param val_file_name: test file name if test data frame is None
        :returns: predicted sentiment
        :rtype: numpy array (1d)

        """
        import review_processing as rp

        if df_test is not None:
            nitems = df_test.shape[0]
            col_names = df_test.columns.values
            if self.review_col_name not in col_names_test:
                sys.exit('predict_ML_model: The name {0} cannot be found'.
                         format(self.review_col_name))
            review_list = df_test[self.review_col_name].values.tolist()
            meaningful_words = map(self.review_to_meaningful_words,
                                   review_list)
        else:
            if test_file_name is None:
                sys.exit('predict_ML_model: test file name does not exist!')
            else:
                suffix = os.path.splitext(test_file_name)[1][1:]
                if suffix == 'csv':
                    df_test = pd.read_csv(self.test_file_name)
                    if self.review_col_name not in col_names:
                        sys.exit('predict_ML_model: The name {0} cannot '
                                 ' be found'.format(self.review_col_name))
                    nitems = df_test.shape[0]
                    review_list = df_test[review_col_name].values.tolist()
                    meaningful_words = map(self.review_to_meaningful_words,
                                           review_list)
                elif suffix == 'json':
                    data_dict_list = rp.load_data(test_file_name)
                    nitems = len(data_dict_list)
                    if self.review_col_name not in data_dict_list.keys():
                        sys.exit('predict_ML_model: The name {0} cannot be '
                                 'found'.format(review_col_name))
                    review_list = map(lambda x: x[review_col_name],
                                      data_dict_list)
                    meaningful_words = map(self.review_to_meaningful_words,
                                           review_list)

        print('The size of test data is {0}'.format(nitems))

        test_data_features = self.vectorizer.transform(meaninful_words)
        test_data_features = test_data_features.toarray()

        sentiment_pred = self.random_forest_model.predict(test_data_features)
        return sentiment_pred

    @staticmethod
    def review_to_meaningful_words(review):
        """Convert review (string) to meaningful words

        :param review: the review string
        :returns: meaningful_words
        :rtype: string, the meaninful_words are separated by ' '

        """

        from nltk.corpus import stopwords

        words = review.split()
        # convert stopwords to a set
        sw_set = set(stopwords.words('english'))
        meaninful_words = [word for word in words if word not in sw_set]

        return meaningful_words
