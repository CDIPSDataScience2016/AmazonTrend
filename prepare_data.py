import pandas as pd
import json
from pprint import pprint

review_elec_fn = '/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/reviews_Electronics.json'
meta_elec_fn = '/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/meta_Electronics.json'
review_vid_fn = '/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/reviews_Video_Games.json'
meta_vid_fn = '/Users/yaningliu/GoogleDrive/DataScience/CDIPS2016/meta_Video_Games.json'

# total number of lines
review_elec_nl = 7824482
meta_elec_nl = 498196
review_vid_nl = 1324753
meta_vid_nl = 50953

review_col_names = ['reviewerID', 'asin', 'reviewerName', 'helpful', \
                    'reviewText', 'overall', 'summary', 'unixReviewTime', \
                    'reviewTime']
meta_col_names = ['asin', 'description', 'price', 'imUrl', 'related', \
                  'salesRank', 'categories', 'buy_after_viewing', 'brand', \
                  'title']


filen = review_vid_fn
with open(filen, 'r') as f:
    data_lines = f.readlines()
print(data_lines[1:5])

df = pd.DataFrame(data_lines)
print(df.head(10))

for i in range(len(data_lines)):
    data_lines[i] = json.loads(data_lines[i])

df = pd.DataFrame(data_lines)
print(df.head(1000))
print(df.columns.values)
