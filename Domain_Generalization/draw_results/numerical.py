import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

font_y = {'family': 'serif',
        #'color':  'darkred',
        'size': 14,
}
data = pd.read_excel('ablations.xlsx', sheet_name = 0)
g = sns.catplot(
    data=data, kind="bar",
    x="k", y="Acc", hue="Norm",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.legend.set_title("")
# plt.ylabel('Weight', fontdict=font_y)
# plt.xlabel('Covariates', fontdict=font_y)
plt.grid(linestyle='--', alpha=0.2)
pdf = PdfPages('weight.pdf')
pdf.savefig()
plt.close()
pdf.close()


# method_list = [ 'TransTEE', 'VCNet', 'TARNet', 'DRNet']
# data = pd.read_excel('params.xlsx', sheet_name = 0)
# ss = data.head()
# print(ss)
# g = sns.catplot(
#     data=data, kind="bar",
#     x="Dataset", y="Log#Params", hue="Method",
#     ci="sd", palette="rocket", height=6
# )
# g.despine(left=True)
# g.legend.set_title("")
# plt.grid(linestyle='--', alpha=0.2)
# pdf = PdfPages('weight.pdf')
# pdf.savefig()
# plt.close()
# pdf.close()
# sns.set_theme(style="white", context="talk")
# f, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
# for i in range(3):
#     data = pd.read_excel('params.xlsx', sheet_name = i)
#     print(data.head())
#     sns.barplot(
#         data=data,
#         x="Dataset", y="Log#Params", hue="Method",
#         ci="sd", palette="rocket",  ax=axs[i]
#     )
#     axs[i].axhline(0, color="k", clip_on=False)
#     if i != 0:
#         axs[i].set(title='', ylabel=None)
#     axs[i].set(xlabel=None)
#     axs[i].legend(loc = 'lower right')
# sns.despine(bottom=True)
# plt.tight_layout(h_pad=-1)
# pdf = PdfPages('weight.pdf')
# pdf.savefig()
# plt.close()
# pdf.close()

# x=[0.012094221077859402, 0.014533461071550846, 0.012580573558807373, 0.014822792261838913, 0.01312377117574215, 0.008466826751828194, 0.012109460309147835, 0.013607637956738472, 0.012299712747335434, 0.03694899380207062, 0.02564762532711029, 0.016815921291708946, 0.0066060470417141914, 0.01105739176273346, 0.010881828144192696, 0.016969801858067513, 0.016906950622797012, 0.021114369854331017, 0.03332620859146118, 0.016029972583055496]

# print('{:.4f} ± {:.4f}'.format(np.mean(x), np.std(x)))

# [[0.00831183 0.07649582 0.0969274  0.02051629 0.02342321 0.03890802
#   0.00200918 0.06374107 0.00733782 0.03720947 0.03867078 0.00612227
#   0.00788331 0.03838759 0.05571199 0.00803067 0.03259746 0.02343803
#   0.02273082 0.07123799 0.05128805 0.0463486  0.07893602 0.07212471
#   0.07161151]] decoder for transtee_tr

# [[0.01259364 0.19020583 0.20893896 0.06333447 0.0104509  0.02223133
#   0.04572909 0.03150243 0.04963088 0.02042547 0.03300935 0.05131306
#   0.02304434 0.0555094  0.01971165 0.01074811 0.02067654 0.01099314
#   0.01943078 0.01526781 0.01140963 0.03117861 0.01748209 0.01425473
#   0.01092777]]decoder for transtee

# [[0.07691569 0.02172426 0.05551937 0.02935646 0.0782875  0.10491341
#   0.05584141 0.02371105 0.04233013 0.02871034 0.01707132 0.02153121
#   0.05518609 0.02985606 0.01992055 0.0550299  0.01878399 0.02181585
#   0.03748923 0.02816603 0.02252991 0.01868958 0.06585941 0.03406043
#   0.03670079]]