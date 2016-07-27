"""This code get the top 10 reviews of a certain product category, and write it as a csv file

Time-stamp: <2016-07-27 12:26:23 yaningliu>
Author: Yaning Liu
"""

import json
from collections import Counter
import operator
import csv
import sys


def get_sorted_product(json_file_name, col_name, write_to_csv=False,
                       write_fn=None, topn=None):
    """Get the sorted product of a certain category in terms of col_name

    :param json_file_name: string, the file name of a json file to read in
    :param col_name: string, the column name to look at, e.g., 'asin'
    :param write_to_csv: bool, if write to csv file
    :param write_fn: string, the name of the csv file to write to
    :param topn: int, get top topn to write
    :returns: dic
    :rtype: a dictionary, with key values the col_name, and values,
    the number of reviews

    """

    name_collection = []
    count = 0
    with open(json_file_name, 'r') as fh:
        for line in fh:
            name_collection.append(json.loads(line)[col_name])
            if count % 100000 == 0:
                print('get_sorted_product: reading record # {}'.format(count),
                      flush=True)
            count += 1

    cnt = dict(Counter(name_collection))

    # Sort the dictionary to a list of tuples
    sorted_name_collection = sorted(cnt.items(), key=operator.itemgetter(1),
                                    reverse=True)

    # get the top topn col_name
    if write_to_csv:
        if write_fn is not None and topn is not None:
            dic_list = []
            key_set = set([tuple[0] for tuple
                           in sorted_name_collection[:topn]])
            count = 0
            with open(json_file_name, 'r') as fh:
                for line in fh:
                    if count % 100000 == 0:
                        print('get_sorted_product: reading record # {}'
                              .format(count), flush=True)
                    dic = json.loads(line)
                    if dic[col_name] in key_set:
                        dic_list.append(dic)
                    if count == 0:
                        keys = list(dic.keys())
                    count += 1

            print(keys)
            with open(write_fn, 'w') as fh:
                dict_writer = csv.DictWriter(fh, keys)
                dict_writer.writeheader()
                dict_writer.writerows(dic_list)
        else:
            sys.exit('get_sorted_product: file name and topn need'
                     ' to be provided!')

    return sorted_name_collection

if __name__ == '__main__':
    tuple_list = get_sorted_product('/Users/yaningliu/GoogleDrive/DataScience/'
                             'CDIPS2016/reviews_Electronics.json', 'asin',
                             True, 'top10_reviews_elec.csv', 10)
    print(tuple_list[:10])
