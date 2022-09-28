import os 
import re
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import json
import pandas as pd
import seaborn as sns
import math
sns.set_theme(style="ticks", palette="pastel")
plt.rcParams['font.family'] = ['simhei']
plt.rcParams["axes.unicode_minus"]=False
matplotlib.rcParams['font.size']=20

names = ['高钾', "铅钡"]
for n in names:
    plt.figure(figsize=(10, 6))#6，8分别对应宽和高

    data = pd.read_excel('exp.xlsx', sheet_name = 0)
    array_data = data.values
    titles = data.columns.values
    X, Y, Z = [], [], []
    for i in range(array_data.shape[0]): 
        if array_data[i][-2] == n:
            X.extend(titles[1:-2])
            Y.extend(array_data[i,1:-2])
            Z.extend([array_data[i][0] for j in range(len(titles)-3)])
    
    for i in range(len(Y)):
        Y[i] = 0 if math.isnan(Y[i]) else Y[i]
        X[i] = X[i].replace('(', '\n(')
    X = np.array(X).reshape((-1,1))
    Y = np.array(Y).reshape((-1,1))
    Z = np.array(Z).reshape((-1,1))
    data = np.concatenate((X,Y,Z), axis=-1)
    df = pd.DataFrame(data)
    df.columns = ["化学成分", "成分所占比例", "风化程度"]
    df["成分所占比例"] = pd.to_numeric(df["成分所占比例"])

    # Draw a nested boxplot to show bills by day and time
    # sns.boxplot(x="化学成分", y="成分所占比例",
    #             hue="风化程度", #palette=["m", "g", ],
    #             data=df)
    # sns.scatterplot(x="化学成分", y="成分所占比例",
    #                 hue="风化程度", linewidth=0, palette="Pastel1",sizes=(1, 8),
    #                 data=df)
    
    sns.boxplot(x="化学成分", y="成分所占比例",
            hue="风化程度", palette=["m", "g", 'b'],
            data=df)
    plt.ylim(0, 20)
    plt.tight_layout(h_pad=-0.1)
    plt.title(n+"玻璃")
    plt.savefig(f'{n}_box.jpg')
    plt.close()