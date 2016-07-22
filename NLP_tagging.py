""" This class deals with tagging (POS, NER) related functions

Time-stamp: <2016-07-21 18:02:14 yaning>

Author: Yaning Liu
Main used modules nltk

"""

import nltk
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
import sys
from collections import Counter

# NER 3 classes
NER_CLASS = ['PERSON', 'ORGANIZATION', 'LOCATION']

# map for POS
map_POS = {'adj': ['JJ', 'JJR', 'JJS'],
           'noun': ['NN', 'NNP', 'MNPS', 'NNS']}

class tagging:

    def __init__(self, NER_model, NER_tagger, POS_model, POS_tagger,
                 use_stanford):
        """The initializer of the class

        :param NER_model: NER model path
        :param NER_tagger: NER tagger path
        :param POS_model: POS model path
        :param POS_tagger: POS tagger path
        :param use_stanford: boolean, if using stanford NER and POS tagging

        """
        self.NER_model = NER_model
        self.NER_tagger = NER_tagger
        self.POS_model = POS_model
        self.POS_tagger = POS_tagger
        self.use_stanford = use_stanford

        if use_stanford:
            self.post = StanfordPOSTagger(self.POS_model, self.POS_tagger)
            self.nert = StanfordNERTagger(self.NER_model, self.NER_tagger)
        else:
            self.post = nltk.pos_tag
            self.nert = StanfordNERTagger(self.NER_model,
                                          self.NER_tagger) # or nltk.ne_chunk

    def show_the_most_frequent_postags(self, dic_list, review_col_name, POS,
                                       product_id=None, id_type=None, topn=5):
        """Given a list of reviews, each review is a list of words,
        and the POS (adj, none),
        show all the word associate with the product and POS and the
        corresponding number of times appeared

        if no product_id and id_type is provided, then the function search
        the whole data for the most frequent words that belong to POS

        :param dic_list: a list of dictionaries, the review of each
        dictionary is a list of words
        :param product_id: string, the product id
        :param id_type: string, the type of the id, e.g. asin
        :param POS: the part of speech, 'adj'/'none' are supported
        :param review_col_name, string the key of the review text value
        :param topn, show the top topn words only
        :returns: a list of list, the internal list is in the form of [word, frequency]
        the list is ordered by frequency in the descending order
        :rtype: a list of [word, frequency]

        """

        if review_col_name not in dic_list[0].keys():
            sys.exit('show_the_most_frequent_postags: The specified key {0} '
                     'can not be found in the dictionaries'
                     .format(review_col_name))

        if ( (product_id is not None and id_type is None) or
             (product_id) is None and id_type is not None):
            sys.exit('show_the_most_frequent_postags: both/neither product_id and id_type'
                    ' should be provided!' .format(id_type))

        if id_type is not None and id_type not in dic_list[0].keys():
            sys.exit('show_the_most_frequent_postags: The specified id type {0} '
                     'can not be found in the dictionaries'
                     .format(id_type))

        if POS not in map_POS.keys():
            sys.exit('show_the_most_frequent_postags: The POS {} '
                     'is not known'.format(POS))

        word_and_freq_dic = []
        if product_id is not None and id_type is not None:
            for dic in dic_list:
                if dic[id_type] == product_id:
                    tagpairs = self.post.tag(dic[review_col_name])
                    for tagpair in tagpairs:
                        if tagpair[1] in map_POS[POS]:
                            word_and_freq_dic.append(tagpair[0])
        elif product_id is None and id_type is None:
            for dic in dic_list:
                tagpairs = self.post.tag(dic[review_col_name])
                for tagpair in tagpairs:
                    if tagpair[1] in map_POS[POS]:
                        word_and_freq_dic.append(tagpair[0])


        word_and_freq_dic = dict(Counter(word_and_freq_dic))

        # sort the dictionary in terms of the values in descending order
        # and return the keys
        keys_ordered = sorted(word_and_freq_dic, key=word_and_freq_dic.get,
                              reverse=True)

        word_and_freq = []
        for key in keys_ordered:
            word_and_freq.append([key, word_and_freq_dic[key]])

        return word_and_freq

    def show_nertags(self, dic_list, review_col_name, NER,
                     product_id, id_type):
        """Given a list of reviews, each review is a list of words,
        and the name class, 'PERSON', 'ORGANIZATION' or 'LOCATION'
        show all named entity of the name classes associated with the product
        corresponding number of times appeared

        :param dic_list: a list of dictionaries, the review of each
        dictionary is a list of words
        :param product_id: string, the product id
        :param id_type: string, the type of the id, e.g. asin
        :param NER: the NER class, 'PERSON', 'ORGANIZATION' or 'LOCATION'
        :param review_col_name, string the key of the review text value
        :returns: a dic, with the NER classes as the keys, and the values
        are a list of pairs in the form of [word, frequency].
        The values are ordered by frequency in the descending order
        :rtype: a dictionary with values as a list of [word, frequency]

        """

        if review_col_name not in dic_list[0].keys():
            sys.exit('show_nertags: The specified key {0} '
                     'can not be found in the dictionaries'
                     .format(review_col_name))

        if product_id is None or id_type is None:
            sys.exit('show_nertags: both product_id and id_type'
                    ' should be provided!')

        for ner in NER:
            if ner not in NER_CLASS:
                sys.exit('show_nertags: The NER {} '
                         'is not known'.format(ner))

        pair_list = []
        if product_id is not None and id_type is not None:
            for dic in dic_list:
                if dic[id_type] == product_id:
                    tagpairs = self.nert.tag(dic[review_col_name])
                    for tagpair in tagpairs:
                        if tagpair[1] in NER_CLASS:
                            pair_list.append(tagpair)

        dic_NER = {}
        for ner in NER_CLASS:
            dic_NER[ner] = []
        for pair in pair_list:
            dic_NER[pair[1]].append(pair[0])

        for key in dic_NER.keys():
            dic_NER[key] = dict(Counter(dic_NER[key]))
            # sort the inner dictionary in terms of the values in
            # descending order and return the keys
            keys_ordered = sorted(dic_NER[key], key=dic_NER[key].get,
                                  reverse=True)
            tmp_list = []
            for key1 in keys_ordered:
                tmp_list.append([key, dic_NER[key][key1]])
                dic_NER[key] = tmp_list

        return dic_NER
