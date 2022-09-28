
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sns.set_theme(style="dark")
# Simulate data from a bivariate Gaussian
n = 5000
loc = 3.0
mean = [loc, loc]
cov = [(1.0, 0), (0, 1.0)]
rng = np.random.RandomState(0)
x, y = rng.multivariate_normal(mean, cov, n).T
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x, y=y, s=5, color="grey")
sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=x, y=y, levels=5, color="grey", linewidths=1)

mean = [0, -loc]
x, y = rng.multivariate_normal(mean, cov, n).T
sns.scatterplot(x=x, y=y, s=5, color="b")
sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako",)
sns.kdeplot(x=x, y=y, levels=5, color="b", linewidths=1)

mean = [-loc, loc]
x, y = rng.multivariate_normal(mean, cov, n).T
sns.scatterplot(x=x, y=y, s=5, color="grey")
sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako", color="y")
sns.kdeplot(x=x, y=y, levels=5, color="grey", linewidths=1)

mean = [-0, -loc]
x, y = rng.multivariate_normal(mean, cov, n).T
sns.scatterplot(x=x, y=y, s=5, color="r")
sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=x, y=y, levels=5, color="r", linewidths=1)

style_dict_interval = {
    '0':dict(color='#dd7e6b'),
    '1':dict(color='#b6d7a8'),
    '2':dict(color='#f9cb9c'),
    '3':dict(color='#a4c2f4'), 
    '4':dict(color='#b4a7d6')
}
val = 2*loc + 1
plt.plot([-val-1, val], [-loc, -loc],  linewidth=1, color='black')
x = np.arange(-2*loc-2, 2*loc + 2)
low_CI_bound = [-2*loc-1 for i in range(len(x))]
high_CI_bound = [-loc for i in range(len(x))]
plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['0'])
low_CI_bound = [-loc for i in range(len(x))]
high_CI_bound = [2*loc+2 for i in range(len(x))]
plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['1'])
# style_dict_interval = {
#     '0':dict(color='#dd7e6b'),
#     '1':dict(color='#b6d7a8'),
#     '2':dict(color='#f9cb9c'),
#     '3':dict(color='#a4c2f4'), 
#     '4':dict(color='#b4a7d6')
# }
# x = np.arange(-2*loc-2, -loc + 1)
# low_CI_bound = [-2*loc-1 for i in range(len(x))]
# high_CI_bound = [2*loc+1 for i in range(len(x))]
# plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['0'])
# x = np.arange(loc, 2*loc + 3)
# low_CI_bound = [-2*loc-1 for i in range(len(x))]
# high_CI_bound = [2*loc+1 for i in range(len(x))]
# plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['0'])
# x = np.arange(-loc, loc + 1)
# low_CI_bound = [-2*loc-1 for i in range(len(x))]
# high_CI_bound = [2*loc+1 for i in range(len(x))]
# plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['1'])

sns.despine(bottom=True)
plt.tight_layout(h_pad=-1)
pdf = PdfPages('weight.pdf')
pdf.savefig()
plt.close()
pdf.close()
def original():
    sns.set_theme(style="dark")
    # Simulate data from a bivariate Gaussian
    n = 5000
    loc = 3.0
    mean = [loc, loc]
    cov = [(1.0, 0), (0, 1.0)]
    rng = np.random.RandomState(0)
    x, y = rng.multivariate_normal(mean, cov, n).T
    f, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=x, y=y, s=5, color="w")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)

    mean = [loc, -loc]
    x, y = rng.multivariate_normal(mean, cov, n).T
    sns.scatterplot(x=x, y=y, s=5, color="b")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako",)
    sns.kdeplot(x=x, y=y, levels=5, color="b", linewidths=1)

    mean = [-loc, loc]
    x, y = rng.multivariate_normal(mean, cov, n).T
    sns.scatterplot(x=x, y=y, s=5, color="y")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako", color="y")
    sns.kdeplot(x=x, y=y, levels=5, color="y", linewidths=1)

    mean = [-loc, -loc]
    x, y = rng.multivariate_normal(mean, cov, n).T
    sns.scatterplot(x=x, y=y, s=5, color="r")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(x=x, y=y, levels=5, color="r", linewidths=1)


    val = 2*loc + 1
    plt.plot([loc, loc], [val, -val], linewidth=1, color='black')
    plt.plot([-loc, -loc], [val, -val], linewidth=1, color='black')

    style_dict_interval = {
        '0':dict(color='#dd7e6b'),
        '1':dict(color='#b6d7a8'),
        '2':dict(color='#f9cb9c'),
        '3':dict(color='#a4c2f4'), 
        '4':dict(color='#b4a7d6')
    }
    x = np.arange(-2*loc-2, -loc + 1)
    low_CI_bound = [-2*loc-1 for i in range(len(x))]
    high_CI_bound = [2*loc+1 for i in range(len(x))]
    plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['0'])
    x = np.arange(loc, 2*loc + 3)
    low_CI_bound = [-2*loc-1 for i in range(len(x))]
    high_CI_bound = [2*loc+1 for i in range(len(x))]
    plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['0'])
    x = np.arange(-loc, loc + 1)
    low_CI_bound = [-2*loc-1 for i in range(len(x))]
    high_CI_bound = [2*loc+1 for i in range(len(x))]
    plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['1'])

    sns.despine(bottom=True)
    plt.tight_layout(h_pad=-1)
    pdf = PdfPages('weight.pdf')
    pdf.savefig()
    plt.close()
    pdf.close()

def align_13():
    sns.set_theme(style="dark")
    # Simulate data from a bivariate Gaussian
    n = 5000
    loc = 3.0
    mean = [loc, loc]
    cov = [(1.0, 0), (0, 1.0)]
    rng = np.random.RandomState(0)
    x, y = rng.multivariate_normal(mean, cov, n).T
    f, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=x, y=y, s=5, color="grey")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(x=x, y=y, levels=5, color="grey", linewidths=1)

    mean = [loc, -loc]
    x, y = rng.multivariate_normal(mean, cov, n).T
    sns.scatterplot(x=x, y=y, s=5, color="grey")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako",)
    sns.kdeplot(x=x, y=y, levels=5, color="grey", linewidths=1)

    mean = [-loc, 0]
    x, y = rng.multivariate_normal(mean, cov, n).T
    sns.scatterplot(x=x, y=y, s=5, color="y")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako", color="y")
    sns.kdeplot(x=x, y=y, levels=5, color="y", linewidths=1)

    mean = [-loc, 0]
    x, y = rng.multivariate_normal(mean, cov, n).T
    sns.scatterplot(x=x, y=y, s=5, color="r")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(x=x, y=y, levels=5, color="r", linewidths=1)


    val = 2*loc + 1
    # plt.plot([loc, loc], [val, -val], linewidth=1, color='black')
    plt.plot([-loc, -loc], [val, -val], linewidth=1, color='black')

    style_dict_interval = {
        '0':dict(color='#dd7e6b'),
        '1':dict(color='#b6d7a8'),
        '2':dict(color='#f9cb9c'),
        '3':dict(color='#a4c2f4'), 
        '4':dict(color='#b4a7d6')
    }
    x = np.arange(-2*loc-2, -loc + 1)
    low_CI_bound = [-2*loc-1 for i in range(len(x))]
    high_CI_bound = [2*loc+1 for i in range(len(x))]
    plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['0'])
    x = np.arange(-loc, 2*loc + 3)
    low_CI_bound = [-2*loc-1 for i in range(len(x))]
    high_CI_bound = [2*loc+1 for i in range(len(x))]
    plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['1'])
    # x = np.arange(loc, 2*loc + 3)
    # plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['0'])
    # x = np.arange(-loc, loc + 1)
    # low_CI_bound = [-2*loc-1 for i in range(len(x))]
    # high_CI_bound = [2*loc+1 for i in range(len(x))]
    # plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.1, **style_dict_interval['1'])

    sns.despine(bottom=True)
    plt.tight_layout(h_pad=-1)
    pdf = PdfPages('weight.pdf')
    pdf.savefig()
    plt.close()
    pdf.close()
