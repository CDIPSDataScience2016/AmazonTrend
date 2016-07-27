""" This class deals with topic modeling.

Time-stamp: <2016-07-27 14:30:29 yaningliu>

Author: Yaning Liu

 The methods considered are:
1) Latent Dirichlet Allocation
2) Latent Semantic Analysis
3) Term frequency inverse document frequency
4) Random Projection
5) Hierarchical Dirichlet Process

Main used modules are gensim
"""

import numpy as np
from gensim import corpora, models, similarities
import logging
import sys
from multiprocessing import Pool
import pandas as pd
import review_processing as rp

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class topic_modeling(object):

    def __init__(self, TM_method, PosNeg='positive', use_pool=False,
                 pool_size=1,
                 save_dict=False, save_dict_name=None,
                 save_corpus=False, save_corpus_name=None,
                 save_topic_model=False, save_topic_model_name=None):
        """initializer for topic modeling class

        :param TM_method: topic modelin method, valid methods are
        LDA(Latent Dirichlet Allocation)
        LSI(Latent Semantic Indexing)
        TFIDF(Term frequency inverse document frequency)
        RP(Random Projections)
        HDP(Hierarchical Dirichlet Process)

        :param PosNeg: 'positive' or 'negative', considering only
        positive or negative reviews
        :param use_pool: boolean, if parallel
        :param pool_size: integer, size of pool
        :param save_dict: if saving dictionary
        :param save_dict_name: the name of the dictioanry if saving
        :param save_corpus: if saving corpus
        :param save_corpus_name: the file name of the copus if saving
        :param save_topic_model: boolean, if saving the topic model
        :param save_topic_model_name: string, the name of the topic model file
        :returns: the class
        :rtype: topic_modeling

        """
        self.TM_method = TM_method
        self.PosNeg = PosNeg
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.save_dict = save_dict
        self.save_dict_name = save_dict_name
        self.save_corpus = save_corpus
        self.save_corpus_name = save_corpus_name
        self.save_topic_model = save_topic_model
        self.save_topic_model_name = save_topic_model_name

    def get_topics(self, dict_list, product_id, id_type, review_col_name,
                   ntopic=10, clean_reviews=False, POS_tagging=False,
                   tagging_obj=None, noun_phrases=None):
        """Get the topics given a list of dictionaries, each list element
        is a cleaned text (a list of words), given the product id and id type
        show the top ntopic topics

        :param dict_list: a list of dictionaries, each element is a list of words
        :param ntopic: show only the top ntopic topics
        :param product_id: string, the product id
        :param id_type: string, the type of the id, e.g. asin
        :param review_col_name, string the key of the review text value
        :returns: one of the topic models (lsi, lda, ...), dictionary
        and corpus
        :rtype: gensim topic model, dictionary and corpus

        """
        if review_col_name not in dict_list[0].keys():
            sys.exit('get_topics: The specified key {0} '
                     'can not be found in the dictionaries'
                     .format(review_col_name))

        if product_id is not None and id_type is None or \
           product_id is None and id_type is not None:
            sys.exit('get_topics: both/neither product_id and id_type'
                     ' should be provided!' .format(id_type))

        if id_type is not None and id_type not in dict_list[0].keys():
            sys.exit('get_topics: The specified id type {0} '
                     'can not be found in the dictionaries'
                     .format(id_type))

        # The criteria for distinguishing postive and negative
        pn_criteria = 'overall'

        # The review list of the product
        if self.PosNeg == 'positive':
            prod_text_list = [dic[review_col_name] for dic in dict_list
                              if dic[id_type] == product_id and
                              dic[pn_criteria] > 3.0]
        elif self.PosNeg == 'negative':
            prod_text_list = [dic[review_col_name] for dic in dict_list
                              if dic[id_type] == product_id and
                              dic[pn_criteria] < 3.0]

        if clean_reviews:
            prod_text_list = [rp.review_processing.
                              review_str_to_wordlist(text, 'BeautifulSoup')
                              for text in prod_text_list]
        if POS_tagging and not noun_phrases:
            prod_text_list = tagging_obj.get_pos_tagged_words(
                prod_text_list, POS=['adj', 'adv', 'noun'])
        elif not POS_tagging and noun_phrases:
            nouns_list = tagging_obj.get_pos_tagged_words(
                prod_text_list, POS=['noun'])
            from textblob import TextBlob
            prod_text_list = [TextBlob(' '.join(text)).noun_phrases
                              for text in prod_text_list]
            for i in range(len(prod_text_list)):
                prod_text_list[i] += nouns_list[i]

            # prod_text_list_tmp = [TextBlob(' '.join(text)).noun_phrases
            #                       for text in prod_text_list]
            # prod_text_list = []
            # for review in prod_text_list_tmp:
            #     tmp = []
            #     for nf in review:
            #         tmp += nf.split()
            #     prod_text_list.append(tmp)

        dictionary = self.build_dictionary(prod_text_list, self.save_dict,
                                           self.save_dict_name)

        corpus = self.get_corpus(dictionary, prod_text_list, self.save_corpus,
                                 self.save_corpus_name, self.use_pool,
                                 self.pool_size)
        topic_model = self.get_topic_model(corpus, self.TM_method,
                                           dictionary, ntopic,
                                           self.save_topic_model,
                                           self.save_topic_model_name)
        return topic_model, dictionary, corpus

    @staticmethod
    def build_dictionary(cleaned_text_list, save_dict=False, save_name=None):
        """build dictionary from cleaned texts

        :param cleaned_text_list: a list of a list of words,
        obtained from cleaned texts. The inner list of words could
        be a single cleaned review
        :param save_dict: bolean, if saving the dictioanry
        :param save_name: string, the file name to save to. If not saving, None
        :returns: dictionary
        :rtype: a gensim dictionary

        """
        dictionary = corpora.Dictionary(cleaned_text_list)
        if save_dict:
            if save_name is None:
                sys.exit('build_dictionary: the file name to save '
                         'to should be provided!')
            else:
                dictionary.save(save_name)

        return dictionary

    @staticmethod
    def get_corpus(dictionary, text_list, save_corpus=False,
                   save_corpus_name=None, use_pool=None, pool_size=1):
        """Compute corpus

        :param dictionary: the gensim dictionary
        :param text_list: a list of texts
        :param save_corpus: boolean if saving the corpus
        :param save_corpus_name: the name of the corpus to save to
        :param use_pool: boolean, if using pool
        :param pool_size: integer, the size of the pool
        :returns: corpus
        :rtype: gensim corpus

        """
        # corpus is a list of bag of words (bow)
        if use_pool:
            pool = Pool(pool_size)
            corpus = pool.map(dictionary.doc2bow, text_list)
            pool.close()
        else:
            corpus = [dictionary.doc2bow(text) for text in text_list]

        if save_corpus:
            # save corpus for later use
            corpora.MmCorpus.serialize(save_corpus_name, corpus)

        return corpus

    @staticmethod
    def get_topic_model(corpus, trans_method, dictionary, ntopics,
                        save_trans_model=False,
                        save_trans_name=None):
        """Obtain a topic model based on one of the transformation method

        :param corpus: the corpus, a list of bag of words
        :param trans_method: the method for transformation
        :param dictionaory: the gensim dictionary
        :param ntopics: number of topics
        :param save_trans_model: boolean if saving transformed
        :param save_trans_name: string, the file name of the
        transformed vector
        :returns: the topic/transformation model
        :rtype:

        """
        if trans_method == 'TFIDF':
            tfidf = models.TfidfModel(corpus)  # initialize a tfidf model
            if save_trans_model:
                if save_trans_name is not None:
                    tfidf.save(save_trans_name)
                else:
                    sys.exit('get_topic_model: no file name specified '
                             'for topic model!')
            return tfidf
        elif trans_method == 'LSI':
            tfidf = models.TfidfModel(corpus)  # initialize a tfidf model
            corpus_tfidf = tfidf[corpus]
            lsi = models.LsiModel(corpus_tfidf, id2word=dictionary,
                                  num_topics=ntopics)
            # corpus_lsi = lsi[corpus_tridf]
            if save_trans_model:
                if save_trans_name is not None:
                    lsi.save(save_trans_name)
                else:
                    sys.exit('get_topic_model: no file name specified '
                             'for topic model!')
            return lsi
        elif trans_method == 'LDA':
            lda = models.LdaModel(corpus, id2word=dictionary,
                                  num_topics=ntopics)
            if save_trans_model:
                if save_trans_name is not None:
                    lda.save(save_trans_name)
                else:
                    sys.exit('get_topic_model: no file name specified '
                             'for topic model!')
            return lda
        elif trans_method == 'HDP':
            hdp = models.HdpModel(corpus, id2word=dictionary)
            if save_trans_model:
                if save_trans_name is not None:
                    hdp.save(save_trans_name)
                else:
                    sys.exit('get_topic_model: no file name specified '
                             'for topic model!')
            return hdp
        elif trans_method == 'RP':
            tfidf = models.TfidfModel(corpus)  # initialize a tfidf model
            corpus_tfidf = tfidf[corpus]
            rp = models.RpModel(corpus_tfidf, num_topics=ntopics)
            if save_trans_model:
                if save_trans_name is not None:
                    rp.save(save_trans_name)
                else:
                    sys.exit('get_topic_model: no file name specified '
                             'for topic model!')
            return rp
        else:
            sys.exit('corpus_transform: topic method {0} is not valid!'
                     .format(trans_method))

    @staticmethod
    def find_similarity(dictionary, corpus, ntopics, doc_to_compare,
                        dictionary_name=None, corpus_name=None,
                        load_index=False, load_index_name=None,
                        save_index=False, save_index_name=None):
        """find the similarity indices for a new document/new documents doc_to_compare

        :param dictionary: the gensim dictionary
        :param corpus: the gensim corpus
        :param ntopics: the number of topics
        :param doc_to_compare: a list of list of words,
        the new document(s) to be compared
        :param dictionary_name: the file name of dictionary
        :param corpus_name: the corpus file name
        :param load_index: boolean, if load index from a file
        :param load_index_name: string, the name of the index file to load from
        :param save_index: boolean, if saving index
        :param save_index_name: string, the file name of index to save to
        :returns: similarity
        :rtype: gensim similarity index, a list of a list of pairs of the form
        (old doc number, similarity)

        """
        if dictionary is None:
            if dictionary_name is None:
                sys.exit('find_similarity: '
                         'dictionary file name has to be provided!')
            else:
                dictionary = corpora.Dictionary.load(dictionary_name)

        if corpus is None:
            if corpus_name is None:
                sys.exit('find_similarity: '
                         'corpus file name has to be provided!')
            else:
                corpus = corpora.MmCorpus(corpus_name)

        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=ntopics)

        if not load_index:
            # transform corpus to LSI space and index it
            index = similarities.MatrixSimilarity(lsi[corpus])
        else:
            if load_index_name is None:
                sys.exit('find_similarity: index file name not provided!')
            else:
                index = similarities.MatrixSimilarity.load(load_index_name)

        if save_index:
            if save_indx_name is None:
                index.save(save_index_name)
            else:
                sys.exit('find_similarity: index file name not provided!')

        sim_list = []
        for doc in doc_to_compare:
            print(doc)
            vec_bow = dictionary.doc2bow(doc)
            vec_lsi = lsi[vec_bow]
            sim = index[vec_lsi]
            sim = sorted(enumerate(sim), key=lambda item: -item[-1])
            sim_list.append(sim)

        return sim_list

    @staticmethod
    def get_product_topic_dataframe(dict_list, product_id, id_type,
                                    review_col_name, model_type='LDA',
                                    clean_reviews=False,
                                    clean_method='regular',
                                    pos_tagging=False, tagging_obj=None,
                                    noun_phrases=None,
                                    load_dictionary=True, dict_file_name=None,
                                    load_corpus=True, corpus_file_name=None,
                                    load_model=True, model_name=None):
        """Get the topics of the product and turn it into a data frame

        :param dict_list: a list of dictionaries
        :param product_id: string, product id
        :param id_type: string, the type of id
        :param review_col_name: the column name of the review
        :param model_type: the type of topic modeling
        :param clean_reviews: boolean, if cleaning reviews
        :param clean_method: the method to clean the reviews.
        'regular', 'POStag', 'ngrams'
        :param pos_tagging: boolean, if pos_tagging. If, tagging, keep only
        adjective, adverbs and nouns
        :param ngram_tagging: if None: not using ngram_tagging. If = 2 use
        2 gram tagging. if = 3, use 3 gram tagging
        :param load_dictionary: bool, if loading dictioanry
        :param dict_file_name: the file name of the dictionary if loading
        :param load_corpus: bool, if loading corpus
        :param corpus_file_name: the file name of the corpus if loading
        :param load_model: bool, if loading the topic model
        :param model_name: the file name of the topic model, if loading
        :returns: df, the topics are sorted in descending order by topic
        probability
        :rtype: a data frame

        """
        if load_dictionary:
            dictionary = corpora.dictionary.Dictionary.load(dict_file_name)
        # if load_corpus:
        #     corpus = corpora.MmCorpus.load(corpus_file_name)
        if load_model and model_type == 'LDA':
            topic_model = models.ldamodel.LdaModel.load(model_name)

        if not clean_reviews:
            review_list = [dic[review_col_name] for dic in dict_list
                           if dic[id_type] == product_id]
        elif clean_reviews:
            raw_reviews = [dic[review_col_name] for dic in dict_list
                           if dic[id_type] == product_id]
            review_list = [rp.review_processing.
                           review_str_to_wordlist(dic[review_col_name])
                           for dic in dict_list if dic[id_type] == product_id]

        if pos_tagging and noun_phrases is None:
            review_list = tagging_obj.get_pos_tagged_words(
                review_list, POS=['adj', 'adv', 'noun'])
        elif noun_phrases and not pos_tagging:
            from textblob import TextBlob
            nouns_list = tagging_obj.get_pos_tagged_words(
                review_list, POS=['noun'])
            review_list = [TextBlob(' '.join(review)).noun_phrases
                           for review in review_list]
            for i in range(len(review_list)):
                review_list[i] += nouns_list[i]

            # review_list_tmp = [TextBlob(' '.join(review)).noun_phrases
            #                    for review in review_list]
            # review_list = []
            # for review in review_list_tmp:
            #     tmp = []
            #     for nf in review:
            #         tmp += nf.split()
            #     review_list.append(tmp)

        nreviews = len(review_list)

        dict_df = {'Review': [], 'Review #': [], 'TopicID': [],
                   'Topic prob': [], 'Words and weights': []}
        # dict_df = {'Review': [], 'TopicID': [],
        #            'Topic prob': [], 'Words and weights': []}
        df = pd.DataFrame()

        for i in range(nreviews):
            bow = dictionary.doc2bow(review_list[i])
            topic_ids = topic_model.get_document_topics(bow)

            TopicID = []
            TopicProb = []
            WordsWeights = []
            for topic_id in topic_ids:
                if not clean_reviews:
                    dict_df['Review'].append(' '.join(review_list[i]))
                elif clean_reviews:
                    dict_df['Review'].append(raw_reviews[i])
                dict_df['Review #'].append(i)

                TopicID.append(topic_id[0])
                TopicProb.append(topic_id[1])
                word_and_probs = topic_model.show_topic(topic_id[0])
                WordsWeights.append('+'.join(['{0:4.2} {1}'
                                              .format(wp[1], wp[0])
                                              for wp in word_and_probs]))

            # sort first
            sort_idx = np.argsort(TopicProb)[::-1]
            TopicID = list(np.array(TopicID)[sort_idx])
            TopicProb = list(np.array(TopicProb)[sort_idx])
            WordsWeights = list(np.array(WordsWeights)[sort_idx])

            idx = np.where(np.array(TopicProb) > 0.8)
            if idx[0].shape[0]:
                TopicID = list(np.array(TopicID)[idx[0]])
                TopicProb = list(np.array(TopicProb)[idx[0]])
                WordsWeights = list(np.array(WordsWeights)[idx[0]])
            else:
                idx = np.where(np.array(TopicProb) > 0.4)
                TopicID = list(np.array(TopicID)[idx])
                TopicProb = list(np.array(TopicProb)[idx])
                WordsWeights = list(np.array(WordsWeights)[idx])

            dict_df['TopicID'].append(TopicID)
            dict_df['Topic prob'].append(TopicProb)
            dict_df['Words and weights'].append('+'.join(['{0:4.2} {1}'.
                                                          format(wp[1], wp[0])
                                                          for wp in
                                                          WordsWeights]))

            if not clean_reviews:
                # iterables = [' '.join(review_list[i]),
                #              np.arange(len(topic_ids))]
                iterables = [i, TopicID]
            if clean_reviews:
                # iterables = [review_list[i], np.arange(len(topic_ids))]
                iterables = [i, TopicID]
            index = pd.MultiIndex.from_product(iterables,
                                               names=['Reviews', 'Topic ID'])
            df_tmp = pd.DataFrame({'Topic prob':
                                   TopicProb,
                                   'Words and weights':
                                   WordsWeights}, index=index)

            df = pd.concat((df, df_tmp))

        # df = pd.DataFrame.from_dict(dict_df)

        return df
