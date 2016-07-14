""" This class deals with processing raw reviews/meta data

Time-stamp: <2016-07-14 14:15:25 yaningliu>

Author: Yaning Liu
Main used modules arebeautifulsoup, pandas

"""

import sys
import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import csv
from multiprocessing import Pool
from itertools import repeat


class review_processing:

    def __init__(self, data_file_name_in, data_file_name_out,
                 col_names_kept, use_pool=False, pool_size=None):
        """Initializer for review_processing

        :param data_file_name_in: the raw data file name string
        :param data_file_name_out: the file name where processed data are
        stored
        :param col_names_kept: list of strings, the column names to be kept
        in the processed data file, the other columns will be abandoned
        :param use_pool: if use multiprocessing Pool
        :param pool_size: the size of the pool
        :returns: the class review_processing
        :rtype: review_processing

        """
        self.data_file_name_in = data_file_name_in
        self.data_file_name_out = data_file_name_out
        self.col_names_kept = col_names_kept
        self.use_pool = use_pool
        self.pool_size = pool_size

    def clean_reviews(self, col_name_clean, nentries=-1,
                      clean_method='BeautifulSoup',
                      remove_numbers=True, remove_punct=True,
                      append_sentiment=False, append_based_on=None,
                      sentiment_col_name=None):
        """Clean all reviews and output the data if

        :param col_name_clean: the name of the column to be cleaned
        :param nentries: specify the number of entries to load, if negative,
        load all entries
        :param clean_method: string, the method for cleaning, e.g.,
        BeautifulSoup
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean if remove punctuations
        :param append_sentiment: boolean, if append sentiment
        :param append_based_on: the keyword based on which sentiment is
        computed
        :param sentiment_col_name: the keyword/column names of sentiment
        :returns: returns clean_reviews
        :rtype: a list of dictionaries, if output_to_file='False'

        """
        print('Cleaning all reviews')
        data_dict_list = self.load_json_data(self.data_file_name_in, nentries)
        if append_sentiment:
            if append_based_on is None or sentiment_col_name is None:
                sys.exit('clean_reviews: append_based_on and/or '
                         'sentiment_col_name have to be provided!')
            self.label_sentiment_from_stars(data_dict_list, append_based_on,
                                            sentiment_col_name)

        col_names = list(data_dict_list[0].keys())
        nitems = len(data_dict_list)
        ncols = len(col_names)

        if col_name_clean in col_names:
            # for loop takes 415 seconds for video review data
            # for i in range(nitems):
            #     if (i+1) % 1000 == 0:
            #         print('Cleaning the item number {0} output of {1} items'.
            #               format(i+1, nitems))
            #     data_dict_list[i][col_name_clean] = self.clean_one_review_str(
            #         data_dict_list[i][col_name_clean], clean_method,
            #         remove_numbers, remove_punct)
            # map takes 415 seconds for video review data
            # data_dict_list = list(map(lambda dic: self.clean_one_review_dict(
            #     dic, clean_method, col_name_clean,
            #     remove_numbers, remove_punct), data_dict_list))
            # list comprehension takes 413 seconds for video review data
            # data_dict_list = [self.clean_one_review_dict(
            #     dic, clean_method, col_name_clean,
            #     remove_numbers, remove_punct) for dic in data_dict_list]

            # Use Multiprocessing
            pool = Pool(4)
            data_dict_list = pool.starmap(self.clean_one_review_dict,
                                          zip(data_dict_list,
                                              repeat(clean_method),
                                              repeat(col_name_clean),
                                              repeat(remove_numbers),
                                              repeat(remove_punct)))
            pool.close()

        else:
            sys.exit(('clean_reviews: The column name for cleaning is '
                      'not found!'))

        # Convert the data to pandas frame
        # df = pd.DataFrame(data_dict_list)
        # print('Data has been prepared into a data frame with {0} items '
        #       'and {1} columns'.format(nitems, ncols))
        #
        # print('The column names are {0}'.format(' '.join(col_names)))

        print('Data has been cleaned. There are {0} entries '
              'and {1} columns'.format(nitems, ncols))
        print('The column names are {0}'.format(' '.join(col_names)))

        for col_name in self.col_names_kept:
            if col_name not in col_names:
                sys.exit('clean_reviews:The name {0} cannot be found'.format
                         (col_name))

        clean_reviews = [{k: v for k, v in iter(dic.items())
                          if k in self.col_names_kept}
                         for dic in data_dict_list]
        return clean_reviews

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
    def clean_one_review_str(dict_with_raw_review, clean_method,
                             remove_numbers=True, remove_punct=True):
        """Clean one single review item (string) and return it as a string

        :param dict_with_raw_review: the unprocessed raw review string
        :param clean_method: the method to clean review, e.g., BeautifulSoup
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean, if remove punctuations
        :returns: cleaned review
        :rtype: string

        """

        if clean_method == 'BeautifulSoup':
            rev_clean = BeautifulSoup(dict_with_raw_review, 'lxml').get_text()
            if remove_numbers and remove_punct:
                rev_clean = re.sub('[^a-zA-Z]', ' ', rev_clean).lower()
            elif remove_numbers and not remove_punct:
                rev_clean = re.sub('[0-9]', ' ', rev_clean).lower()
            elif not remove_numbers and remove_punct:
                rev_clean = re.sub('[^a-zA-Z0-9]', ' ', rev_clean).lower()
            else:
                rev_clean = rev_clean.lower()
        else:
            sys.exit(('clean_one_review_str: The clean method not '
                      'supported yet!'))

        return rev_clean

    @staticmethod
    def clean_one_review_dict(dic, clean_method,
                              col_name_clean,
                              remove_numbers=True, remove_punct=True):
        """Clean the review value in a single one dict and return the cleaned

        :param dic: the dict with unprocessed raw review
        :param clean_method: the method to clean review, e.g., BeautifulSoup
        :param col_name_clean: the name of the column to be cleaned
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean, if remove punctuations
        :returns: , the dict with cleaned review
        :rtype: string

        """
        col_names = dic.keys()
        if col_name_clean not in col_names:
            sys.exit('clean_one_review_dict: keys {0} not found!'.
                     format(col_name_clean))

        if clean_method == 'BeautifulSoup':
            dic[col_name_clean] = BeautifulSoup(
                dic[col_name_clean], 'lxml').get_text()
            if remove_numbers and remove_punct:
                dic[col_name_clean] = re.sub('[^a-zA-Z]', ' ',
                                             dic[col_name_clean]).lower()
            elif remove_numbers and not remove_punct:
                dic[col_name_clean] = re.sub('[0-9]', ' ',
                                             dic[col_name_clean]).lower()
            elif not remove_numbers and remove_punct:
                dic[col_name_clean] = re.sub('[^a-zA-Z0-9]', ' ',
                                             dic[col_name_clean]).lower()
            else:
                dic[col_name_clean] = dic[col_name_clean].lower()
        else:
            sys.exit(('clean_one_review_dict: The clean method not '
                      'supported yet!'))

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
            dict[sentiment_col_name] = 1 if dict[append_based_on] > 3 else -1
