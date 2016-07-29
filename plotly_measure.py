# Get this figure: fig = py.get_figure("https://plot.ly/~HawleDap/15/")
# Get this figure's data: data = py.get_figure("https://plot.ly/~HawleDap/15/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="Density vs Portion of bar", fileopt="extend"))
# Get y data of first trace: y1 = py.get_figure("https://plot.ly/~HawleDap/15/").get_data()[0]["y"]

# Get figure documentation: https://plot.ly/python/get-requests/
# Add data documentation: https://plot.ly/python/file-options/

# If you're using unicode in your file, you may need to specify the encoding.
# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

import json
import plotly
import plotly.plotly as py
# from plotly.graph_objs import *
import plotly.graph_objs as go
import model_validation_data as mvd
import numpy as np

py.sign_in('naddata', '6eos5rv0q4')

N = 8
xx = np.linspace(0, 1, N)
key_list = list(mvd.measure_data.keys())

# Create traces
trace0 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[0]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[0]
)
trace1 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[1]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[1]
)
trace2 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[2]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[2]
)
trace3 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[3]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[3]
)
trace4 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[4]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[4]
)
trace5 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[5]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[5]
)
trace6 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[2]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[6]
)
trace7 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[3]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[7]
)
trace8 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[4]]['roc_auc'],
    mode = 'lines+markers',
    name = key_list[8]
)
data_auc = [trace0, trace1, trace2, trace3, trace4,
            trace5, trace6, trace7, trace8]

# Plot and embed in ipython notebook!
# py.iplot(data_auc, filename='auc_plotly')

layout = go.Layout(title='AUC measures for different traing data size',
                   xaxis=dict(
                       title='Proportion of training data used'
                   ),
                   yaxis=dict(
                       title='AUC',
                   ),
                   showlegend=True)
fig = go.Figure(data=data_auc, layout=layout)
fig_js = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
with open('auc_plotly.json', 'w') as fh:
    fh.write(fig_js)


# -----------------------------------------------------
# Create traces
trace0 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[0]]['f1'],
    mode = 'lines+markers',
    name = key_list[0]
)
trace1 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[1]]['f1'],
    mode = 'lines+markers',
    name = key_list[1]
)
trace2 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[2]]['f1'],
    mode = 'lines+markers',
    name = key_list[2]
)
trace3 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[3]]['f1'],
    mode = 'lines_markers',
    name = key_list[3]
)
trace4 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[4]]['f1'],
    mode = 'lines+markers',
    name = key_list[4]
)
trace5 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[5]]['f1'],
    mode = 'lines+markers',
    name = key_list[5]
)
trace6 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[2]]['f1'],
    mode = 'lines+markers',
    name = key_list[6]
)
trace7 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[3]]['f1'],
    mode = 'lines+markers',
    name = key_list[7]
)
trace8 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[4]]['f1'],
    mode = 'lines+markers',
    name = key_list[8]
)
data_f1 = [trace0, trace1, trace2, trace3, trace4,
        trace5, trace6, trace7, trace8]

# Plot and embed in ipython notebook!
# py.iplot(data_f1, filename='f1_plotly')

layout = go.Layout(title='F1 measures for different traing data size',
                   xaxis=dict(
                       title='Proportion of training data used'
                   ),
                   yaxis=dict(
                       title='F1',
                   ),
                   showlegend=True)
fig = go.Figure(data=data_f1, layout=layout)
fig_js = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
with open('f1_plotly.json', 'w') as fh:
    fh.write(fig_js)


# ----------------------------------------------
# -----------------------------------------------------
# Create traces
trace0 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[0]]['accuracy'],
    mode = 'lines+markers',
    name = key_list[0]
)
trace1 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[1]]['accuracy'],
    mode = 'lines+markers',
    name = key_list[1]
)
trace2 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[2]]['accuracy'],
    mode = 'lines+markers',
    name = key_list[2]
)
trace3 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[3]]['accuracy'],
    mode = 'lines_markers',
    name = key_list[3]
)
trace4 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[4]]['accuracy'],
    mode = 'lines+markers',
    name = key_list[4]
)
trace5 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[5]]['accuracy'],
    mode = 'lines+markers',
    name = key_list[5]
)
trace6 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[2]]['accuracy'],
    mode = 'lines+markers',
    name = key_list[6]
)
trace7 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[3]]['accuracy'],
    mode = 'lines+markers',
    name = key_list[7]
)
trace8 = go.Scatter(
    x = xx,
    y = mvd.measure_data[key_list[4]]['accuracy'],
    mode = 'lines+markers',
    name = key_list[8]
)
data_accuracy = [trace0, trace1, trace2, trace3, trace4,
        trace5, trace6, trace7, trace8]

# Plot and embed in ipython notebook!
# py.iplot(data_accuracy, filename='accuracy_plotly')

layout = go.Layout(title='Accuracy measures for different traing data size',
                   xaxis=dict(
                       title='Proportion of training data used'
                   ),
                   yaxis=dict(
                       title='Accuracy',
                   ),
                   showlegend=True)
fig = go.Figure(data=data_accuracy, layout=layout)
fig_js = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
with open('accuracy_plotly.json', 'w') as fh:
    fh.write(fig_js)
