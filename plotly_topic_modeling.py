"""This module plot plots for topic modeling with plotly
Time-stamp: <2016-07-29 23:48:14 yaningliu>

Author: Yaning Liu
"""

import json
import pandas as pd
import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
# from app import df_review, df_topic

asin_dic = {'B0074BW614': 'Kindle Fire',
            'B00DR0PDNE': 'Google Chromecast',
            'B007WTAJTO': 'SanDisk memory card',
            'B006GWO5WK': 'Kindle Powerfast Charger',
            'B007R5YDYA': 'Kindle Paperwhite Case',
            'B00622AG6S': 'Powergen Car Charger',
            'B008OHNZI0': 'Privacy Screen for iPhone 5',
            'B009SYZ8OC': 'USB to lightning charger',
            'B00BGA9WK2': 'Sony PlayStation 4',
            'B004QK7HI8': 'Mohu Leaf 30 TV Antenna'}

inv_asin_dic = {k: v for v, k in asin_dic.items()}


def plotly_topic_frequency_bar(product_id, sentiment):
    """Making topic-review frequency charts and topic-review tables
    with plotly

    :param product_id: String, the name of the product, the names are the key
    values of the asin_dic
    :param sentiment: string, 'positive' or 'negative'
    :returns: fig_json, html_table_str
    :rtype: a json string containing the plotly json script for the frequencey
    chart, and an html string containing the topic-review table

    """

    py.sign_in('naddata', '6eos5rv0q4')

    nshow = 50
    threshold = 0.5

    # df_review = pd.read_csv('../data/review-topic.csv')
    # df_topic = pd.read_csv('../data/topic-words-reviews.csv')

    df_review = pd.read_csv('review-topic.csv')
    df_topic = pd.read_csv('topic-words-reviews.csv')

    # product_id = 'B0074BW614'

    if sentiment == 'positive':
        df = df_topic[df_topic['Sentiment'] == 1]
    elif sentiment == 'negative':
        df = df_topic[df_topic['Sentiment'] == 0]
    # elif sentiment == 'all':
    #     df = df_topic

    TID = df.Topic_ID.tolist()
    # TID_label = ['Topic ' + str(i) for i in TID]
    rev_prob = [eval(rev) for rev in df.Reviews.tolist()]
    rev_prob = [[freq for freq in freq1 if freq[1] > threshold
                 and df_review.loc[freq[0], 'ProductID'] ==
                 inv_asin_dic[product_id]]
                for freq1 in rev_prob]
    frequency = [len(freq) for freq in rev_prob]
    sort_idx = np.argsort(np.array(frequency))
    frequency_sorted = list(np.array(frequency)[sort_idx])
    # TID_label_sorted = list(np.array(TID_label)[sort_idx])
    TID_sorted = list(np.array(TID)[sort_idx])

    if sentiment == 'positive':
        ylabel = [', '.join([pair[0] for pair in
                             eval(df_topic.loc[TID, 'Words_and_Weights'])[:5]])
                  for TID in TID_sorted]
    elif sentiment == 'negative':
        ylabel = [', '.join([pair[0] for pair in
                             eval(df_topic.loc[TID+200,
                                               'Words_and_Weights'])[:5]])
                  for TID in TID_sorted]

    layout = go.Layout(title='Frequency of reviews related to each topic '
                       'for product ' + product_id,
                       xaxis=dict(
                           title='Review frequency'
                       ),
                       # yaxis=dict(
                       #     title='Topics'
                       # ),
                       showlegend=False,
                       margin=go.Margin(l=400),
                       height=800)

    data = [go.Bar(x=frequency_sorted[-nshow:],
                   # y=TID_label_sorted[-nshow:],
                   y=ylabel[-nshow:],
                   orientation='h')]

    fig = go.Figure(data=data, layout=layout)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # make topic, review, probability table
    ntopic_in_tab = 5
    nreview_in_tab = 10
    if sentiment == 'positive':
        reviews = [[df_review.loc[pair[0], 'Reviews'] for pair in
                    eval(df_topic.loc[TID, 'Reviews'])[:nreview_in_tab]]
                   for TID in TID_sorted[-ntopic_in_tab:]]
        probs = [[pair[1] for pair in
                 eval(df_topic.loc[TID, 'Reviews'])[:nreview_in_tab]
                  if pair[1] > threshold]
                 for TID in TID_sorted[-ntopic_in_tab:]]
    elif sentiment == 'negative':
        reviews = [[df_review.loc[pair[0], 'Reviews'] for pair in
                    eval(df_topic.loc[TID+200, 'Reviews'])[:nreview_in_tab]
                    if pair[1] > threshold]
                   for TID in TID_sorted[-ntopic_in_tab:]]
        probs = [[pair[1] for pair in
                 eval(df_topic.loc[TID+200, 'Reviews'])[:nreview_in_tab]
                  if pair[1] > threshold]
                 for TID in TID_sorted[-ntopic_in_tab:]]

    df = pd.DataFrame()

    for i in range(len(probs)):
        df_tmp = pd.DataFrame({'Topic': ylabel[-ntopic_in_tab+i],
                               'Reviews': reviews[-ntopic_in_tab+i],
                               'Prob': probs[-ntopic_in_tab+i]}) \
                   .set_index(['Topic', 'Reviews'])
        df = pd.concat((df, df_tmp))

    # table = FF.create_table(df, index=True)
    # py.iplot(table, filename='pandas_table')

    # df.to_html('table_html.html')
    html_table_str = df.to_html()
    # with open('topic_freq.json', 'w') as fh:
    #    fh.write(fig_js)

    return fig_json, html_table_str

    # with open('topic_freq.json', 'w') as fh:
    #    fh.write(fig_js)
