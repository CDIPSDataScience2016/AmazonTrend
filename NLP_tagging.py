""" This class deals with tagging (POS, NER) related functions

Time-stamp: <2016-07-19 19:24:28 yaning>

Author: Yaning Liu
Main used modules nltk

"""

import nltk
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
import sys
from collections import Counter

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

    def show_POS_by_product(self, dic_list, product_id,
                            id_type, POS, review_col_name):
        """Given a list of reviews, each review is a list of words,
        and the product id and id type, and the POS (adj, adv,...),
        show all the word associate with the product and POS and the
        corresponding number of times appeared

        :param dic_list: a list of dictionaries, the review of each
        dictionary is a list of words
        :param product_id: string, the product id
        :param id_type: string, the type of the id, e.g. asin
        :param POS: the part of speech, for example
        :param review_col_name, string the key of the review text value
        :returns: a dictionary with word as the keys and the frequency as the
        values
        :rtype: a list of strings

        """

        if review_col_name not in dic_list[0].keys():
            sys.exit('show_POS_by_product: The specified key {0} '
                     'can not be found in the dictionaries'
                     .format(review_col_name))

        if id_type not in dic_list[0].keys():
            sys.exit('show_POS_by_product: The specified id type {0} '
                     'can not be found in the dictionaries'
                     .format(id_type))

        word_and_freq = []
        for dic in dic_list:
            if dic[id_type] == product_id:
                tagpairs = self.post.tag(dic[review_col_name])
                for tagpair in tagpairs:
                    if tagpair[1] == POS:
                        wordlist.append(tagpair[0])

        word_and_freq = dict(Counter(word_and_freq))

        return word_and_freq
