import matplotlib.pyplot as plt
import model_validation_data as mvd
import numpy as np
import plotly.tools as tls
import plotly.plotly as plpl

markers = ['o', 'v', '^', '*', 's', 'd', '+', 'p', '1']
colors = ['b', 'r', 'c', 'g', 'm', 'k', 'y', 'crimson', 'maroon']
linestyles = [':', '-.', '--', '-', ':', '-.', '--', '-', ':']
save_names = ['roc_auc', 'f1', 'accuracy']
fs = 10

# ROC_AUC
roc_auc = plt.figure()
plt.title('roc_auc measures for different traing data size')
plt.xlabel('Proportion of training data used')
plt.ylabel('roc_auc')

X_proportion = np.linspace(0.1, 1.0, 8)
counter = 0
for key in mvd.measure_data.keys():
    plt.plot(X_proportion, mvd.measure_data[key]['roc_auc'],
             c=colors[counter], ls=linestyles[counter],
             marker=markers[counter], label=key)
    counter += 1
plt.legend(loc='best', fontsize=fs)
# plt.savefig(save_names[0]+'.png')


# F1
f1 = plt.figure()
plt.title('f1 measures for different traing data size')
plt.xlabel('Proportion of training data used')
plt.ylabel('f1')

X_proportion = np.linspace(0.1, 1.0, 8)
counter = 0
for key in mvd.measure_data.keys():
    plt.plot(X_proportion, mvd.measure_data[key]['f1'],
             c=colors[counter], ls=linestyles[counter],
             marker=markers[counter], label=key)
    counter += 1
plt.legend(loc='best', fontsize=fs)
# plt.savefig(save_names[1]+'.png')

# Accuracy
acc = plt.figure()
plt.title('Accuracy measures for different traing data size')
plt.xlabel('Proportion of training data used')
plt.ylabel('accuracy')

X_proportion = np.linspace(0.1, 1.0, 8)
counter = 0
for key in mvd.measure_data.keys():
    plt.plot(X_proportion, mvd.measure_data[key]['accuracy'],
             c=colors[counter], ls=linestyles[counter],
             marker=markers[counter], label=key)
    counter += 1
plt.legend(loc='best', fontsize=fs)
# plt.savefig(save_names[2]+'.png')

# plotly_roc_auc = tls.mpl_to_plotly(roc_auc)
# plotly_f1 = tls.mpl_to_plotly(f1)
# plotly_acc = tls.mpl_to_plotly(acc)
#
# url_roc_auc = plpl.plot(plotly_roc_auc)
# url_f1 = plpl.plot(plotly_f1)
# url_acc = plpl.plot(plotly_acc)
# url_roc_auc = plpl.plot_mpl(roc_auc, filename='roc_auc_plotly')
# url_f1 = plpl.plot_mpl(f1, filename='f1_plotly')
# url_acc = plpl.plot_mpl(acc, filename='acc_plotly')
