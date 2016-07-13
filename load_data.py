import json
import pandas as pd

review_file = 'reviews_Electronics.json'
meta_file = 'meta_Electronics.json'

def fix_json(jstr):
    if jstr[0] == jstr[-1] == "'":
        return jstr[1:-1]
    return jstr

review_file = open(review_file,'r')
#meta_lines = open(meta_file,'r')

data = []
count = 0
while True:
    if count > 10000:
        break
    try:
        line = review_file.readline()
    except:
        break
    count += 1
    data.append(json.loads(line))

#json.dump(data,'reviews.json')

