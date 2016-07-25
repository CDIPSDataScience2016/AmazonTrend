"""This is a class for training a Word2Vec/Doce2Vec model based on our data

Time-stamp: <2016-07-21 23:57:24 yaning>
Main used modules are nltk, gensim, beautifulsoup
"""

import numpy as np
from multiprocessing import Pool
import logging
from gensim.models import word2vec
import nltk.data
import review_processing as rp
import sys
from itertools import repeat

from nltk.corpus import stopwords
from gensim.models.doc2vec import LabeledSentence
from gensim.models import doc2vec
from gensim import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class my_word2vec:
    """
    Word2Vec class
    """

    def __init__(self, review_col_name,
                 tokenizer, use_pool=False, pool_size=None,
                 output_model=False, output_model_name=None,
                 remove_stopwords=False, remove_numbers=True,
                 remove_punct=True, clean_method='BeautifulSoup',
                 num_features=300,
                 min_word_count=40, context=10,
                 downsampling=1e-3):
        """The initializer of the word2vec_train class

        :param review_col_name: string the column name of review texts
        :param tokenizer: the tokenizer
        :param use_pool: boolean, if using pool
        :param pool_size: int, the size of the pool
        :param output_model: boolean, if outputing the model to file
        :param output_model_name: the name of the output file
        :param remove_stopwords: boolean, if removing stopwords
        :param remove_numbers: boolean if remove numbers
        if output_model=true
        :param remove_punct: boolean if remove punctuations
        :param clean_method: string, the cleaning method
        :param num_features: integer, the number of features
        :param min_word_count: integer
        :param context: integer
        :param downsampling: float
        :returns: a word2vec model
        :rtype: word2vec model

        """

        self.review_col_name = review_col_name
        self.tokenizer = tokenizer
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.output_model = output_model
        self.output_model_name = output_model_name
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.remove_punct = remove_punct
        self.clean_method = clean_method
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.context = context
        self.downsampling = downsampling

    def train_word2vec_model(self, sentences):
        """Train a word2vec model

        :param sentences: a list of sentences (a sentence is a list of words )
        :returns: word2vec model
        :rtype: word2vec

        """

        logging.info('Training word2vec model...')
        model = word2vec.Word2Vec(sentences, workers=self.pool_size,
                                  size=self.num_features,
                                  min_count=self.min_word_count,
                                  window=self.context,
                                  sample=self.downsampling)

        if self.output_model:
            if self.output_model_name is None:
                sys.exit('train_word2vec_model: output_model_name not provided!')
            else:
                model.save(self.output_model_name)

        return model

    def reviews_to_sentences(self, review_list):
        """For a list of reviews, clean each review, and tranform
        each review to a list of sentences (each sentence is a list of words).
        Finally add all the list to a single list, each elment of the list
        is a sentence

        :param review_list: a list of reviews
        :returns: the list of all sentences from all reviews
        :rtype: a list of list of words (string)

        """

        if self.use_pool:
            pool = Pool(self.pool_size)
            sentences_tmp = pool.starmap(
                self.review_to_sentences_static,
                zip(review_list, repeat(self.tokenizer),
                    repeat(self.clean_method), repeat(self.remove_numbers),
                    repeat(self.remove_punct), repeat(self.remove_stopwords)))
            pool.close()

            sentences = []
            for sentence in sentences_tmp:
                sentences.extend(sentence)
        else:
            sentences = []
            for review in review_list:
                sentences += self.review_to_sentences(review)

        return sentences


    def review_to_sentences(self, review):
        """For a single review, clean each review, and tranform
        it to a list of sentences (each sentence is a list of words).

        :param review: a single review
        :returns: the list of all sentences from the reviews
        :rtype: a list of list of words (string)

        """

        # Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = self.tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(rp.review_processing.
                                 review_str_to_wordlist(raw_sentence,
                                                        self.clean_method,
                                                        self.remove_numbers,
                                                        self.remove_punct,
                                                        self.remove_stopwords))
        return sentences

    @staticmethod
    def review_to_sentences_static(review, tokenizer, clean_method,
                                   remove_numbers, remove_punct,
                                   remove_stopwords):
        """For a single review, clean each review, and tranform
        it to a list of sentences (each sentence is a list of words).

        :param review: a single review
        :param tokenizer: the tokenizer
        :param remove_stopwords: boolean, if removing stopwords
        :param remove_numbers: boolean if remove numbers
        if output_model=true
        :param remove_punct: boolean if remove punctuations
        :param clean_method: string, the cleaning method
        :returns: the list of all sentences from the reviews
        :rtype: a list of list of words (string)

        """

        # Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(rp.review_processing.
                                 review_str_to_wordlist(raw_sentence,
                                                        clean_method,
                                                        remove_numbers,
                                                        remove_punct,
                                                        remove_stopwords))
        return sentences

class my_doc2vec:

    def __init__(self, review_col_name,
                 tokenizer, use_pool=False, pool_size=None,
                 output_model=False, output_model_name=None,
                 remove_stopwords=False, remove_numbers=True,
                 remove_punct=True, clean_method='BeautifulSoup',
                 num_features=300, nepoch = 10,
                 min_word_count=40, context=10,
                 downsampling=1e-3):
        """The initializer of the doc2vec_train class

        :param review_col_name: string the column name of review texts
        :param tokenizer: the tokenizer
        :param use_pool: boolean, if using pool
        :param pool_size: int, the size of the pool
        :param output_model: boolean, if outputing the model to file
        :param output_model_name: the name of the output file
        :param remove_stopwords: boolean, if removing stopwords
        :param remove_numbers: boolean if remove numbers
        if output_model=true
        :param remove_punct: boolean if remove punctuations
        :param clean_method: string, the cleaning method
        :param num_features: integer, the number of features
        :param nepoch: the number of epoch to train
        :param min_word_count: integer
        :param context: integer
        :param downsampling: float
        :returns: a doc2vec model
        :rtype: doc2vec model

        """

        self.review_col_name = review_col_name
        self.tokenizer = tokenizer
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.output_model = output_model
        self.output_model_name = output_model_name
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.remove_punct = remove_punct
        self.clean_method = clean_method
        self.num_features = num_features
        self.nepoch = nepoch
        self.min_word_count = min_word_count
        self.context = context
        self.downsampling = downsampling

    def train_doc2vec_model(self, sentences_in):
        """Train a doc2vec model

        :param sentences: a list of sentences (a sentence is a list of words )
        :returns: doc2vec model
        :rtype: doc2vec

        """

        logging.info('Training doc2vec model...')

        model = doc2vec.Doc2Vec(workers=self.pool_size,
                                size=self.num_features,
                                min_count=self.min_word_count,
                                window=self.context,
                                sample=self.downsampling)
        sentences = LabeledListOfSentence(sentences_in)
        model.build_vocab(sentences)
        for epoch in range(self.nepoch):
            print('Training epoch {}'.format(epoch))
            sys.stdout.flush()
            model.train(sentences)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate

        if self.output_model:
            if self.output_model_name is None:
                sys.exit('train_doc2vec_model: output_model_name not provided!')
            else:
                model.save(self.output_model_name)

        return model

    def reviews_to_sentences(self, review_list):
        """For a list of reviews, clean each review, and tranform
        each review to a list of sentences (each sentence is a list of words).
        Finally add all the list to a single list, each elment of the list
        is a sentence

        :param review_list: a list of reviews
        :returns: the list of all sentences from all reviews
        :rtype: a list of list of words (string)

        """

        pool = Pool(self.pool_size)
        sentences_tmp = pool.starmap(
            self.review_to_sentences_static,
            zip(review_list, repeat(self.tokenizer),
                repeat(self.clean_method), repeat(self.remove_numbers),
                repeat(self.remove_punct), repeat(self.remove_stopwords)))
        pool.close()

        sentences = []
        for sentence in sentences_tmp:
            sentences.extend(sentence)

        # sentences = []
        # for review in review_list:
        #     sentences += self.review_to_sentences(review)

        return sentences


    def review_to_sentences(self, review):
        """For a single review, clean each review, and tranform
        it to a list of sentences (each sentence is a list of words).

        :param review: a single review
        :returns: the list of all sentences from the reviews
        :rtype: a list of list of words (string)

        """

        # Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = self.tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(rp.review_processing.
                                 review_str_to_wordlist(raw_sentence,
                                                        self.clean_method,
                                                        self.remove_numbers,
                                                        self.remove_punct,
                                                        self.remove_stopwords))
        return sentences

    @staticmethod
    def review_to_sentences_static(review, tokenizer, clean_method,
                                   remove_numbers, remove_punct,
                                   remove_stopwords):
        """For a single review, clean each review, and tranform
        it to a list of sentences (each sentence is a list of words).

        :param review: a single review
        :param tokenizer: the tokenizer
        :param remove_stopwords: boolean, if removing stopwords
        :param remove_numbers: boolean if remove numbers
        if output_model=true
        :param remove_punct: boolean if remove punctuations
        :param clean_method: string, the cleaning method
        :returns: the list of all sentences from the reviews
        :rtype: a list of list of words (string)

        """

        # Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(rp.review_processing.
                                 review_str_to_wordlist(raw_sentence,
                                                        clean_method,
                                                        remove_numbers,
                                                        remove_punct,
                                                        remove_stopwords))
        return sentences

# Iterator class for labeling a list of sentences,
# each of the sentence is itself a list of words
class LabeledListOfSentence(object):
    def __init__(self, review_list):
        self.sentences = review_list
    def __iter__(self):
        for uid, wordlist in enumerate(self.sentences):
            yield LabeledSentence(words=wordlist,
                                  tags=['SENT_%s' % uid])
