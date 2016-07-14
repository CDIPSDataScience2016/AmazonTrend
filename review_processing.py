""" This class deals with processing raw reviews/meta data

Time-stamp: <2016-07-13 17:05:47 yaningliu>

Author: Yaning Liu
Main used modules arebeautifulsoup, pandas

"""

import sys
import json
import pandas as pd
from bs4 import BeautifulSoup
import re


class review_processing:

    def __init__(self, data_file_name_in, data_file_name_out,
                 col_names_kept):
        """Initializer for review_processing

        :param data_file_name_in: the raw data file name string
        :param data_file_name_out: the file name where processed data are
        stored
        :param col_names_kept: list of strings, the column names to be kept
        in the processed data file, the other columns will be abandoned
        :returns: the class review_processing
        :rtype: review_processing

        """
        self.data_file_name_in = data_file_name_in
        self.data_file_name_out = data_file_name_out
        self.col_names_kept = col_names_kept

    def clean_reviews(self, col_name_clean, clean_method='BeautifulSoup',
                      remove_numbers=True, remove_punct=True,
                      output_to_file=False, output_file_type=None,
                      output_file_name=None):
        """Clean all reviews and output the data if
        output_to_file=True

        :param col_name_clean: the name of the column to be cleaned
        :param clean_method: string, the method for cleaning, e.g.,
        BeautifulSoup
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean if remove punctuations
        :param output_to_file: Boolean. If True, output_file_type and
        output_file_name has to be provided
        :param output_file_type: string, the file type of output, e.g., 'csv'
        :param output_file_name: string, the output file name
        :returns: if output_to_file=True, returns nothing;
        otherwise returns df_clean_reviews, a pandas data frame
        :rtype: pandas data frame, if output_to_file='False'

        """

        print('Cleaning all reviews')
        data_dict_list = self.load_data(self.data_file_name_in)
        df = pd.DataFrame(data_dict_list)

        print('Data has been prepared into a data frame with {0} items '
              'and {1} columns'.format(df.shape[0], df.shape[1]))

        col_names = df.columns.values
        print('The column names are {0}'.format(' '.join(col_names)))

        if col_name_clean in col_names:
            for i in range(100):
                if (i+1) % 10 == 0:
                    print('Cleaning the item number {0} output of {1} items'.
                          format(i+1, df.shape[0]))
                df.loc[i,col_name_clean] = self.clean_one_review(
                    df.loc[i, col_name_clean], clean_method, remove_numbers,
                    remove_punct)
        else:
            sys.exit(('clean_reviews: The column name for cleaning is '
                      'not found!'))

        for col_name in self.col_names_kept:
            if col_name not in col_names:
                sys.exit('clean_reviews:The name {0} cannot be found'.format
                         (col_name))

        df_clean_reviews = df[self.col_names_kept]

        if output_to_file:
            if output_file_type is None or output_file_name is None:
                sys.exit('clean_reviews: Output type and/or filenames have '
                         'to be provided')
            else:
                if output_file_type == 'csv':
                    pd.DataFrame.to_csv(output_file_name)
                elif output_file_type == 'json':
                    pd.DataFrame.to_json(output_file_name)
                else:
                    sys.exit('clean_reviews: Output type not supported yet!')
        else:
            return df_clean_reviews

    @staticmethod
    def load_data(data_file_name_in):
        """Load the review data to a list of strings by readlines and

        :param data_file_name_in: the raw data file name string
        :returns: data_lines, the processed reviews
        :rtype: list of dictionaries

        """
        print('Loading data to a list of dictionaries')

        with open(data_file_name_in, 'r') as fin:
            data_lines = fin.readlines()

        len_data = len(data_lines)

        for i in range(len_data):
            data_lines[i] = json.loads(data_lines[i])

        print('Loading data finished')
        return data_lines

    @staticmethod
    def clean_one_review(raw_review, clean_method,
                         remove_numbers=True, remove_punct=True):
        """Clean one single review item and return it as a

        :param raw_review: the unprocessed raw review string
        :param clean_method: the method to clean review, e.g., BeautifulSoup
        :param remove_numbers: boolean if remove numbers
        :param remove_punct: boolean, if remove punctuations
        :returns: cleaned review
        :rtype: string

        """

        if clean_method == 'BeautifulSoup':
            rev_clean = BeautifulSoup(raw_review, 'lxml').get_text()
            if remove_numbers and remove_punct:
                rev_clean = re.sub('[^a-zA-Z]', ' ', rev_clean).lower()
            elif remove_numbers and not remove_punct:
                rev_clean = re.sub('[0-9]', ' ', rev_clean).lower()
            elif not remove_numbers and remove_punct:
                rev_clean = re.sub('[^a-zA-Z0-9]', ' ', rev_clean).lower()
            else:
                rev_clean = rev_clean.lower()
        else:
            sys.exit(('clean_one_review: The clean method not '
                      'supported yet!'))

        return rev_clean
