import os 
import re
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

style_dict = {
    '0':dict(linestyle='-', marker='o',markersize=0.1,color='#dd7e6b'),
    '1':dict(linestyle='-',marker='*',markersize=0.1,color='#b6d7a8'),
    '2':dict(linestyle='-',marker='s',markersize=0.1,color='#f9cb9c'),
    '3':dict(linestyle='-',marker='v',markersize=0.1,color='#a4c2f4'), 
    '4':dict(linestyle='-',marker='+',markersize=0.1,color='#b4a7d6')
}

style_dict_interval = {
    '0':dict(color='#dd7e6b'),
    '1':dict(color='#b6d7a8'),
    '2':dict(color='#f9cb9c'),
    '3':dict(color='#a4c2f4'), 
    '4':dict(color='#b4a7d6')
}

plt.style.use(['light','grid'])
font_y = {'family': 'serif',
        #'color':  'darkred',
        #'size': 20,
}
font_x = {'family': 'serif',
        #'size': 20,
}

means = np.array([[90.1, 93.4, 94.0, 94.6, 94.5, 96.4],
[92.4, 95.0,95.5,95.9,96.1,96.4],
[84.4, 84.4,84.4,84.4,84.4,84.4],
[86.5,86.5,86.5,86.5,86.5,86.5]])
stds = np.array([[0.5, 0.4, 0.3, 0.2, 0.3, 0.2],
[0.3, 0.1, 0.1,0.1,0.1, 0.2],
[0.1,0.1,0.1,0.1,0.1,0.1],
[0.4,0.4,0.4,0.4,0.4,0.4]])
def draw(means, stds, name):
    x = [1, 2, 3, 4, 5, 7]
    values = ['1', '2', '3', '4','5','∞'] 
    for a in range(4):
        y_mean, y_std = means[a], stds[a]
        low_CI_bound, high_CI_bound = y_mean - y_std, y_mean + y_std

        plt.plot(x, y_mean, **style_dict[str(a)])
        plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.24, **style_dict_interval[str(a)])
    plt.xticks(x,values)
    plt.ylabel('Acc', fontdict=font_y)
    plt.xlabel('k', fontdict=font_y)
    #plt.xlabel('Treatment Selection Bias', fontdict=font_y)
    plt.legend((r'Power w/ norm', 'Power wo/ norm', 'Exp w/ norm', 'Exp wo/ norm'), frameon=False, loc='best')
    pdf = PdfPages(name)
    #plt.ylim(0, 10)
    #plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()
draw(means, stds, 'ablation_rmnist0.pdf')