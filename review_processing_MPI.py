""" processes raw reviews/meta data with MPI

Time-stamp: <2016-07-17 13:21:09 yaningliu>

Author: Yaning Liu

"""

from mpi4py import MPI
import review_processing as rp
import sys
import time
import json
import nltk


# -------------- Parameters -------------------------
machine = 'LRC'   # can be LRC or MAC or CORI
# ---------------------------------------------------

# -----------------------fixed parameters------------------------------
if machine == 'MAC':
    review_elec_fn = ('/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/'
                      'reviews_Electronics.json')
    meta_elec_fn = ('/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/'
                    'meta_Electronics.json')
    review_vid_fn = ('/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/'
                     'reviews_Video_Games.json')
    meta_vid_fn = ('/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/'
                   'meta_Video_Games.json')

    review_elec_clean_fn = ('/Users/yaningliu/GoogleDrive/DataScience/'
                            'CDIPS2016/reviews_Electronics_clean')
    meta_elec_clean_fn = ('/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/'
                          'meta_Electronics_clean')
    review_vid_clean_fn = ('/Users/yaningliu/GoogleDrive/DataScience/'
                           'CDIPS2016/reviews_Video_Games_clean')
    meta_vid_clean_fn = ('/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/'
                         'meta_Video_Games_clean')
elif machine == 'LRC':
    review_elec_fn = ('/clusterfs/lawrencium/yaningl/DataScience/CDIPS2016/'
                      'reviews_Electronics.json')
    meta_elec_fn = ('/clusterfs/lawrencium/yaningl/DataScience/CDIPS2016/'
                    'meta_Electronics.json')
    review_vid_fn = ('/clusterfs/lawrencium/yaningl/DataScience/CDIPS2016/'
                     'reviews_Video_Games.json')
    meta_vid_fn = ('/clusterfs/lawrencium/yaningl/DataScience/CDIPS2016/'
                   'meta_Video_Games.json')

    review_elec_clean_fn = ('/clusterfs/lawrencium/yaningl/DataScience/'
                            'CDIPS2016/reviews_Electronics_clean')
    meta_elec_clean_fn = ('/clusterfs/lawrencium/yaningl/DataScience/'
                          'CDIPS2016/meta_Electronics_clean')
    review_vid_clean_fn = ('/clusterfs/lawrencium/yaningl/DataScience/'
                           'CDIPS2016/reviews_Video_Games_clean')
    meta_vid_clean_fn = ('/clusterfs/lawrencium/yaningl/DataScience/'
                         'CDIPS2016/meta_Video_Games_clean')

    nltk.data.path.append('/clusterfs/lawrencium/yaningl/scilib/nltk_data')
elif machine == 'CORI':
    review_elec_fn = ('/global/cscratch1/sd/yaning/DataScience/CDIPS2016/'
                      'reviews_Electronics.json')
    meta_elec_fn = ('/global/cscratch1/sd/yaning/DataScience/CDIPS2016/'
                    'meta_Electronics.json')
    review_vid_fn = ('/global/cscratch1/sd/yaning/DataScience/CDIPS2016/'
                     'reviews_Video_Games.json')
    meta_vid_fn = ('/global/cscratch1/sd/yaning/DataScience/CDIPS2016/'
                   'meta_Video_Games.json')

    review_elec_clean_fn = ('/global/cscratch1/sd/yaning/DataScience/'
                            'CDIPS2016/reviews_Electronics_clean')
    meta_elec_clean_fn = ('/global/cscratch1/sd/yaning/DataScience/'
                          'CDIPS2016/meta_Electronics_clean')
    review_vid_clean_fn = ('/global/cscratch1/sd/yaning/DataScience/'
                           'CDIPS2016/reviews_Video_Games_clean')
    meta_vid_clean_fn = ('/global/cscratch1/sd/yaning/DataScience/'
                         'CDIPS2016/meta_Video_Games_clean')

    nltk.data.path.append('/global/cscratch1/sd/yaning/scilib/nltk_data')

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

if rank == 0:
    for i in range(1, nproc):
        print('Rank 0 is obtaining data from rank {0}'.format(i))
        review_tmp = comm.recv(source=i)
        dict_list_local += review_tmp
    t1 = time.time()

    print('Distributed data have been cleaned and  sent back to rank 0. '
          'Elapsed time is {0}'.format(t1-t0))
    print('The number of total items is {0}'.
          format(len(dict_list_local)))
    sys.stdout.flush()

if rank == 0:
    t0 = time.time()
    print('rank {0}: Saving data to json file '.format(rank))
    sys.stdout.flush()
    for i in range(len(dict_list_local)):
        if i == 0:
            with open(data_file_name_out+'.json', 'w') as fout:
                json.dump(dict_list_local[0], fout)
                fout.write('\n')
            fout.close()
        else:
            with open(data_file_name_out+'.json', 'a') as fout:
                json.dump(dict_list_local[i], fout)
                fout.write('\n')
    t1 = time.time()
    print('rank {0}: Saving data to json file finished. Elapsed time is {1}'
          .format(rank, t1-t0))
    sys.stdout.flush()
