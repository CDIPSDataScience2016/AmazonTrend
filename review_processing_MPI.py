
""" processes raw reviews/meta data with MPI

Time-stamp: <2016-07-21 11:10:37 yaning>

Author: Yaning Liu

"""

from mpi4py import MPI
import review_processing as rp
import sys
import time
import json
import nltk


# -------------- Parameters ----------------------------
machine = 'Edison'   # can be LRC or MAC or Cori or Edison
# ------------------------------------------------------

# -----------------------fixed parameters------------------------------
if machine == 'MAC':
    prefix = '/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/'
    prefix = '/Users/yaningliu/nltk_data/'

    review_elec_fn = prefix + 'reviews_Electronics.json'
    meta_elec_fn = prefix + 'meta_Electronics.json'
    review_vid_fn = prefix + 'reviews_Video_Games.json'
    meta_vid_fn = prefix + 'meta_Video_Games.json'

    review_elec_clean_fn = prefix + 'reviews_Electronics_clean'
    review_elec_clean_fn_sent = prefix + 'reviews_Electronics_clean_sent'
    meta_elec_clean_fn = prefix + 'meta_Electronics_clean'
    review_vid_clean_fn = prefix + 'reviews_Video_Games_clean'
    review_vid_clean_fn_sent = prefix + 'reviews_Video_Games_clean_sent'
    meta_vid_clean_fn = prefix + 'meta_Video_Games_clean'
    word2vec_model_vid_fn = prefix + 'word2vec_vid_model'
    word2vec_model_elec_fn = prefix + 'word2vec_elec_model'

    nltk.data.path.append(prefix_nltk)
    vid_w2v_model_name = prefix + 'review_vid_w2v'
    elec_w2v_model_name = prefix + 'review_elec_w2v'
    vid_d2v_model_name = prefix + 'review_vid_d2v'
    elec_d2v_model_name = prefix + 'CDIPS2016/review_elec_d2v'

    stanford_POS_model = prefix_nltk + 'taggers/stanford-postagger-full-2015-12-09/models/english-bidirectional-distsim.tagger'
    stanford_POS_tagger = prefix_nltk + 'taggers/stanford-postagger-full-2015-12-09/stanford-postagger.jar'
    stanford_NER_model = prefix_nltk + 'taggers/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz'
    stanford_NER_tagger = prefix_nltk + 'taggers/stanford-ner-2015-12-09/stanford-ner.jar'

elif machine == 'LRC':
    prefix = '/clusterfs/lawrencium/yaningl/DataScience/CDIPS2016/'
    prefix_nltk = '/clusterfs/lawrencium/yaningl/scilib/nltk_data/'

    review_elec_fn = prefix + 'reviews_Electronics.json'
    meta_elec_fn = prefix + 'meta_Electronics.json'
    review_vid_fn = prefix + 'reviews_Video_Games.json'
    meta_vid_fn = prefix + 'meta_Video_Games.json'

    review_elec_clean_fn = prefix + 'reviews_Electronics_clean'
    review_elec_clean_fn_sent = prefix + 'reviews_Electronics_clean_sent'
    meta_elec_clean_fn = prefix + 'meta_Electronics_clean'
    review_vid_clean_fn = prefix + 'reviews_Video_Games_clean'
    review_vid_clean_fn_sent = prefix + 'reviews_Video_Games_clean_sent'
    meta_vid_clean_fn = prefix + 'meta_Video_Games_clean'
    word2vec_model_vid_fn = prefix + 'word2vec_vid_model'
    word2vec_model_elec_fn = prefix + 'word2vec_elec_model'

    nltk.data.path.append(prefix_nltk)
    vid_w2v_model_name = prefix + 'review_vid_w2v'
    elec_w2v_model_name = prefix + 'review_elec_w2v'
    vid_d2v_model_name = prefix + 'review_vid_d2v'
    elec_d2v_model_name = prefix + 'review_elec_d2v'

    stanford_POS_model = prefix_nltk + 'taggers/stanford-postagger-full-2015-12-09/models/english-bidirectional-distsim.tagger'
    stanford_POS_tagger = prefix_nltk + 'taggers/stanford-postagger-full-2015-12-09/stanford-postagger.jar'
    stanford_NER_model = prefix_nltk + 'taggers/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz'
    stanford_NER_tagger = prefix_nltk + 'taggers/stanford-ner-2015-12-09/stanford-ner.jar'

elif machine == 'Cori':
    prefix = '/global/cscratch1/sd/yaning/DataScience/CDIPS2016/'
    prefix_nltk = '/global/cscratch1/sd/yaning/scilib/nltk_data/'

    review_elec_fn = prefix + 'reviews_Electronics.json'
    meta_elec_fn = prefix + 'meta_Electronics.json'
    review_vid_fn = prefix + 'reviews_Video_Games.json'
    meta_vid_fn = prefix + 'meta_Video_Games.json'

    review_elec_clean_fn = prefix + 'reviews_Electronics_clean'
    review_elec_clean_fn_sent = prefix + 'reviews_Electronics_clean_sent'
    meta_elec_clean_fn = prefix + 'meta_Electronics_clean'
    review_vid_clean_fn = prefix + 'reviews_Video_Games_clean'
    review_vid_clean_fn_sent = prefix + 'reviews_Video_Games_clean_sent'
    meta_vid_clean_fn = prefix + 'meta_Video_Games_clean'
    word2vec_model_vid_fn = prefix + 'word2vec_vid_model'
    word2vec_model_elec_fn = prefix + 'word2vec_elec_model'

    nltk.data.path.append(prefix_nltk)
    vid_w2v_model_name = prefix + 'review_vid_w2v'
    elec_w2v_model_name = prefix + 'review_elec_w2v'
    vid_d2v_model_name = prefix + 'review_vid_d2v'
    elec_d2v_model_name = prefix + 'review_elec_d2v'

    stanford_POS_model = prefix_nltk + 'taggers/stanford-postagger-full-2015-12-09/models/english-bidirectional-distsim.tagger'
    stanford_POS_tagger = prefix_nltk + 'taggers/stanford-postagger-full-2015-12-09/stanford-postagger.jar'
    stanford_NER_model = prefix_nltk + 'taggers/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz'
    stanford_NER_tagger = prefix_nltk + 'taggers/stanford-ner-2015-12-09/stanford-ner.jar'

elif machine == 'Edison':
    prefix = '/scratch1/scratchdirs/yaning/DataScience/CDIPS2016/'
    prefix_nltk = '/scratch1/scratchdirs/yaning/scilib/nltk_data/'

    review_elec_fn = prefix + 'reviews_Electronics.json'
    meta_elec_fn = prefix + 'meta_Electronics.json'
    review_vid_fn = prefix + 'reviews_Video_Games.json'
    meta_vid_fn = prefix + 'meta_Video_Games.json'

    review_elec_clean_fn = prefix + 'reviews_Electronics_clean'
    review_elec_clean_fn_sent = prefix + 'reviews_Electronics_clean_sent'
    meta_elec_clean_fn = prefix + 'meta_Electronics_clean'
    review_vid_clean_fn = prefix + 'reviews_Video_Games_clean'
    review_vid_clean_fn_sent = prefix + 'reviews_Video_Games_clean_sent'
    meta_vid_clean_fn = prefix + 'meta_Video_Games_clean'
    word2vec_model_vid_fn = prefix + 'word2vec_vid_model'
    word2vec_model_elec_fn = prefix + 'word2vec_elec_model'

    nltk.data.path.append(prefix_nltk)
    vid_w2v_model_name = prefix + 'review_vid_w2v'
    elec_w2v_model_name = prefix + 'review_elec_w2v'
    vid_d2v_model_name = prefix + 'review_vid_d2v'
    elec_d2v_model_name = prefix + 'review_elec_d2v'

    stanford_POS_model = prefix_nltk + 'taggers/stanford-postagger-full-2015-12-09/models/english-bidirectional-distsim.tagger'
    stanford_POS_tagger = prefix_nltk + 'taggers/stanford-postagger-full-2015-12-09/stanford-postagger.jar'
    stanford_NER_model = prefix_nltk + 'taggers/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz'
    stanford_NER_tagger = prefix_nltk + 'taggers/stanford-ner-2015-12-09/stanford-ner.jar'

review_col_names = ['reviewerID', 'asin', 'reviewerName', 'helpful',
                    'reviewText', 'overall', 'summary', 'unixReviewTime',
                    'reviewTime']
meta_col_names = ['asin', 'description', 'price', 'imUrl', 'related',
                  'salesRank', 'categories', 'buy_after_viewing', 'brand',
                  'title']

review_elec_nl = 7824482
meta_elec_nl = 498196
review_vid_nl = 1324753
meta_vid_nl = 50953
# -----------------------fixed parameters------------------------------

# -------------- Parameters -------------------------
# col_names_kept = ['helpful', 'reviewText', 'overall', 'reviewTime',
#                   'sentiment']
# col_names_kept = ['reviewText', 'overall', 'sentiment']
col_names_kept = review_col_names + ['sentiment']
col_name_clean_in = 'reviewText'
data_file_name_in = review_elec_fn
data_file_name_out = review_elec_clean_fn
use_pool_in = True
pool_size_in = 16
NLP_model_in = 'BagOfWords'
ML_method_in = 'LogisticRegression'  # 'RandomForest'
review_col_name_in = 'reviewText'
sentiment_col_name_in = 'sentiment'
append_based_on_in = 'overall'
remove_stopwords_in = True
RF_n_est = 100  # random forest number of estimator
load_data_num = -1
test_size_in = 0.5
# -------------- Parameters -------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
if rank == 0:
    print('Number of processors = ' + str(nproc))

if rank == 0:
    t0 = time.time()
    dict_list = rp.review_processing.load_json_data(data_file_name_in,
                                                    load_data_num)
    t1 = time.time()
    print('Elapsed time for loading {0} using load_json_data = {1}'.
          format(data_file_name_in, t1-t0))
    sys.stdout.flush()

if rank == 0:
    nitems = len(dict_list)
    for i in range(1, nproc):
        comm.send(nitems, dest=i)
if rank != 0:
    nitems = comm.recv(source=0)
# comm.bcast(nitems, root=0)  # bcast for some reason doesn't work

comm.Barrier()

if rank != nproc-1:
    local_assignment = list(range(rank*nitems//nproc,
                                  (rank+1)*nitems//nproc))
else:
    local_assignment = list(range((nproc-1)*nitems//nproc, nitems))

print('rank ' + str(rank) + ' has ' +
      str(len(local_assignment)) + ' items to process')
sys.stdout.flush()

if rank == 0:
    t0 = time.time()
    print('rank {0}: Sending data to the other ranks'.format(rank))
    sys.stdout.flush()
    dict_list_local = dict_list[rank*nitems//nproc:
                                (rank+1)*nitems//nproc]
    for j in range(1, nproc-1):
        comm.send(dict_list[j*nitems//nproc:
                                      (j+1)*nitems//nproc], dest=j)
    # for the last processor
    comm.send(dict_list[(nproc-1)*nitems//nproc:nitems],
                        dest=nproc-1)

if rank != 0:
    dict_list_local = comm.recv(source=0)

comm.Barrier()

if rank == 0:
    t1 = time.time()
    print('rank {0}: Data have been sent and received by the other ranks'
          .format(rank))
    sys.stdout.flush()
    print('Elapsed time for distributing and receiving data is {}'.
          format(t1-t0))
    sys.stdout.flush()
    del dict_list

rp_local = rp.review_processing()
cleaned_reviews_local = rp_local.clean_reviews(
    dict_list_local, col_name_clean=col_name_clean_in,
    append_sentiment=True,
    append_based_on=append_based_on_in,
    sentiment_col_name=sentiment_col_name_in)

comm.Barrier()
if rank == 0:
    print('All ranks have finished cleaning!')

t0 = time.time()
if rank != 0:
    comm.send(cleaned_reviews_local, dest=0)

# Rank 0 obtain data and write to file
if rank == 0:
    with open(data_file_name_out+'.json', 'w') as fout:
        pass

    print('Rank 0 is saving data of itself')
    sys.stdout.flush()
    with open(data_file_name_out+'.json', 'a') as fout:
        for i in range(len(cleaned_reviews_local)):
            json.dump(cleaned_reviews_local[i], fout)
            fout.write('\n')
    for i in range(1, nproc):
        print('Rank 0 is obtaining data from rank {0}'.format(i))
        review_tmp = comm.recv(source=i)
        print('Rank 0 is saving data of from rank {0}'.format(i))
        sys.stdout.flush()
        with open(data_file_name_out+'.json', 'a') as fout:
            for j in range(len(review_tmp)):
                json.dump(review_tmp[j], fout)
                fout.write('\n')
    t1 = time.time()

    print('rank {0}: Obtaining and Saving data to json file finished. '
          'Elapsed time is {1}'.format(rank, t1-t0))
    sys.stdout.flush()
