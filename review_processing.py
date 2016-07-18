""" This class deals with processing raw reviews/meta data

Time-stamp: <2016-07-17 11:04:19 yaningliu>

Author: Yaning Liu
Main used modules arebeautifulsoup, pandas

"""

import sys
import json
from bs4 import BeautifulSoup
import re
import csv
from multiprocessing import Pool
from itertools import repeat


class review_processing:

    def __init__(self, use_pool=False, pool_size=None):
        """Initializer for review_processing

        :param use_pool: if use multiprocessing Pool
        :param pool_size: the size of the pool
        :returns: the class review_processing
        :rtype: review_processing

        """

        self.use_pool = use_pool
        self.pool_size = pool_size

    def clean_reviews(self, dict_list, col_name_clean,
                      clean_method='BeautifulSoup',
                      remove_numbers=True, remove_punct=True,
                      remove_stopwords=True, append_sentiment=False,
                      append_based_on=None, sentiment_col_name=None):
        """Clean all reviews and output the data

        :param dict_list: the list of dictionaries containing the data
        :param col_name_clean: the name of the column to be cleaned
        :param clean_method: string, the method for cleaning, e.g.,
        BeautifulSoup
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean if remove punctuations
        :param remove_stopwords: boolean, if remove stopwords
        :param append_sentiment: boolean, if append sentiment
        :param append_based_on: the keyword based on which sentiment is
        computed
        :param sentiment_col_name: the keyword/column names of sentiment
        :returns: returns clean_reviews
        :rtype: a list of dictionaries, if output_to_file='False'

        """
        print('Cleaning all reviews')
        nitems = len(dict_list)
        if append_sentiment:
            if append_based_on is None or sentiment_col_name is None:
                sys.exit('clean_reviews: append_based_on and/or '
                         'sentiment_col_name have to be provided!')
            self.label_sentiment_from_stars(dict_list,
                                            append_based_on,
                                            sentiment_col_name)

        col_names = list(dict_list[0].keys())
        ncols = len(col_names)

        if col_name_clean in col_names:
            if not self.use_pool:
                # for loop takes 415 seconds for video review data
                for i in range(nitems):
                    if (i+1) % 1000 == 0:
                        print('Cleaning the item number {0} output of {1} '
                              'items'.format(i+1, nitems))
                    dict_list[i][
                        col_name_clean] = self.review_str_to_wordlist(
                            dict_list[i][col_name_clean], clean_method,
                            remove_numbers, remove_punct, remove_stopwords)
                # # map takes 415 seconds for video review data
                # dict_list = list(map(lambda
                #                           dic: self.review_dic_to_wordlist(
                #                               dic, clean_method,
                #                               col_name_clean,
                #                               remove_numbers, remove_punct,),
                #                           dict_list))
                # # list comprehension takes 413 seconds for video review data
                # dict_list = [self.review_dic_to_wordlist(
                #     dic, clean_method, col_name_clean,
                #     remove_numbers, remove_punct) for dic in dict_list]
            if self.use_pool:
                # Use Multiprocessing
                # 220s for video review data
                pool = Pool(self.pool_size)
                dict_list = pool.starmap(self.review_dic_to_wordlist,
                                         zip(dict_list,
                                             repeat(clean_method),
                                             repeat(col_name_clean),
                                             repeat(remove_numbers),
                                             repeat(remove_punct),
                                             repeat(remove_stopwords)))
                pool.close()
                # Use joblib
                # dict_list = Parallel(n_jobs=self.pool_size)(
                #     delayed(self.review_dic_to_wordlist)(dic)
                #     for dic in dict_list)

        else:
            sys.exit(('clean_reviews: The column name for cleaning is '
                      'not found!'))

        # Convert the data to pandas frame
        # df = pd.DataFrame(dict_list)
        # print('Data has been prepared into a data frame with {0} items '
        #       'and {1} columns'.format(nitems, ncols))
        #
        # print('The column names are {0}'.format(' '.join(col_names)))

        print('Data has been cleaned. There are {0} entries '
              'and {1} columns'.format(nitems, ncols))
        print('The column names are {0}'.format(' '.join(col_names)))
        sys.stdout.flush()

        return dict_list

    @staticmethod
    def load_json_data(data_file_name_in, nentries=-1):
        """Load the review data to a list of strings by readlines and

        :param data_file_name_in: the raw data file name string
        :param nentries: specify the number of entries to load, if negative,
        load all entries
        :returns: data_lines, the processed reviews
        :rtype: list of dictionaries

        """
        print('Loading data to a list of dictionaries')

        if nentries < 0:
            with open(data_file_name_in, 'r') as fin:
                data_lines = fin.readlines()
                len_data = len(data_lines)

                for i in range(len_data):
                    data_lines[i] = json.loads(data_lines[i])
        else:
            data_lines = []
            with open(data_file_name_in, 'r') as fin:
                for i in range(nentries):
                    # readline gives a dictionary directly, instead of a string
                    data_lines.append(json.loads(fin.readline()))

        print('Loading data finished')
        return data_lines

    @staticmethod
    def write_dict_data(data, file_name_out, file_type, keys):
        """Write the review data (list of dicts) to a file

        :param data: the list of dicts to be written to files
        :param file_name_out: the output data file name string
        :param file_type: the type of the file, e.g., 'csv' or 'json'
        :param keys: the keys correponding to which the data will be written
        :returns: None
        :rtype: None

        """
        print('Writing data to a list of dictionaries')
        if file_type == 'csv':
            with open(file_name_out, 'w') as fout:
                writer = csv.DictWriter(fout, keys)
                writer.writeheader()
                writer.writerows(data)
        elif file_type == 'json':
            with open(file_name_out, 'w') as fout:
                # This writes the list of dictionaries to
                # a single line in the file
                json.dump(data, fout)
        else:
            sys.exit('write_dict_data: Output type not supported yet!')

        print('Writing data finished')

    @staticmethod
    def remove_columns(dict_list, col_names_kept):
        """Remove columns not tin col_name_kept

        :param dict_list: the list of dictionaries
        :param col_name_kept: the column names to be kept (list of strings)
        :returns: clean_reviews
        :rtype: list of dictionaries

        """

        col_names = list(dict_list[0].keys())
        for col_name in col_names_kept:
            if col_name not in col_names:
                sys.exit('remove_columns: The name {0} cannot be found'.format
                         (col_name))

        clean_reviews = [{k: v for k, v in iter(dic.items())
                          if k in col_names_kept}
                         for dic in dict_list]
        return clean_reviews

    @staticmethod
    def review_str_to_sentences(review, clean_method,
                                remove_numbers=True, remove_punct=True,
                                remove_stopwords=True):
        """Transform one single review item (string) and return it as a list of
        sentences, each sentence is a list of words.

        :param review: the unprocessed raw review string
        :param clean_method: the method to clean review, e.g., BeautifulSoup
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean, if remove punctuations
        :param remove_stopwords: boolean, if remove stopwords
        :returns: cleaned sentences
        :rtype: a list of list of words (string)

        """

        from nltk.corpus import stopwords
        import nltk.data
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(review_processing.
                                 review_str_to_wordlist(raw_sentence,
                                                        clean_method,
                                                        remove_numbers,
                                                        remove_punct,
                                                        remove_stopwords))
        return sentences

    @staticmethod
    def review_str_to_wordlist(raw_review, clean_method,
                               remove_numbers=True, remove_punct=True,
                               remove_stopwords=True):
        """Clean one single review item (string) and return it as a list of
        words

        :param raw_review: the unprocessed raw review string
        :param clean_method: the method to clean review, e.g., BeautifulSoup
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean, if remove punctuations
        :param remove_stopwords: boolean, if remove stopwords
        :returns: cleaned reviews,
        :rtype: string

        """

        from nltk.corpus import stopwords

        if clean_method == 'BeautifulSoup':
            word_list = BeautifulSoup(raw_review, 'lxml').get_text()
        else:
            sys.exit(('review_str_to_wordlist: The clean method not '
                      'supported yet!'))

        if remove_numbers and remove_punct:
            word_list = re.sub('[^a-zA-Z]', ' ', word_list).lower().split()
        elif remove_numbers and not remove_punct:
            word_list = re.sub('[0-9]', ' ', word_list).lower().split()
        elif not remove_numbers and remove_punct:
            word_list = re.sub('[^a-zA-Z0-9]', ' ', word_list).lower().split()
        else:
            word_list = word_list.lower().split()

        if remove_stopwords:
            stops = set(stopwords.words('english'))
            word_list = [word for word in word_list if word not in stops]

        return word_list

    @staticmethod
    def review_dic_to_wordlist(dic, clean_method, col_name_clean,
                               remove_numbers=True, remove_punct=True,
                               remove_stopwords=True):
        """Clean the review value in a single one dict and return the cleaned
        dic
        :param dic: the dict with unprocessed raw review
        :param clean_method: the method to clean review, e.g., BeautifulSoup
        :param col_name_clean: the name of the column to be cleaned
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean, if remove punctuations
        :param remove_stopwords: boolean, if remove stopwords
        :returns: , the dict with cleaned review
        :rtype: dict

        """
        col_names = dic.keys()
        if col_name_clean not in col_names:
            sys.exit('review_dic_to_wordlist: keys {0} not found!'.
                     format(col_name_clean))

        dic[col_name_clean] = review_processing.review_str_to_wordlist(
            dic[col_name_clean], clean_method, remove_numbers, remove_punct,
            remove_stopwords
        )
        return dic

    @staticmethod
    def label_sentiment_from_stars(data_dict_list, append_based_on='stars',
                                   sentiment_col_name='sentiment'):
        """Append a keyword with name sentiment_col_name and its value to each
        dictionary of a list of dictionaries. The values are obtained based on
        the values corresponding to keyword append_based_on. The purpose is to
        label the data based on stars/overall if sentiment is not labeled

        :param data_dict_list: a list of dictionaries
        :param append_based_on: the column name/keyword of stars/oveall
        :param sentiment_col_name: the column name/keyword of sentiment
        :returns: data_dict_list
        :rtype: a list of dictionaries

        """

        for dict in data_dict_list:
            dict[sentiment_col_name] = 1 if dict[append_based_on] > 3 else 0
