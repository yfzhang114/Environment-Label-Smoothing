import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import MultipleLocator

dir_path = 'visulizations/PACS'
# paths = ["datas/dann-step5", "datas/dann-step5-eps0.5"]
paths = ["dann_grad/DANNPACS00", "dann_grad_dstep5_eps05/DANNPACS00"]#, '5wlr1e-2alpha0.1-lambda1.00PACS', '5wlr1e-2alpha0.01-lambda1.00PACS']
plt.rcParams["font.family"] = "Times New Roman"
colors = ['gold', 'grey', '#d7191c', 'blue', '#2b83ba']
method_names = ['DANN', 'DANN+ELS']
matplotlib.rcParams.update({'font.size': 18})
plt_content = ['Acc of the target domain', 'Avg Acc of source domains','Classification Loss', 'Discrimination Loss']
plt_names = ['acc_target.pdf', 'acc_source.pdf', 'cls_loss.pdf', 'lossd.pdf']
file_names = ['avg_acc_source.npy', 'avg_acc_target.npy', 'losses_e.npy', 'norm.npy']
plt_content = ['Sum of Gradients']
plt_names = ['norm.pdf']
file_names = ['norm.npy']

for i in range(len(plt_content)):
    # if 'PACS' in paths[0]:
    #     if i == 1:
    #         plt.ylim(0.88, 0.98)
    #     elif i == 0:
    #         plt.ylim(0.92, 0.99)
    #     elif i == 3:
    #         plt.ylim(0, 10)
    #     iterval = 100
    # else:
    #     if i == 1:
    #         plt.ylim(0.95, 1.0)
    #     elif i == 0:
    #         plt.ylim(0.95, 1.0)
    #     elif i == 3:
    #         plt.ylim(0, 10)
    for j in range(0, len(paths)):
        content_path = os.path.join(dir_path, os.path.join(paths[j], file_names[i]))
        content = np.load(content_path)
        x = np.arange(content.shape[0])
        # if i < 2:
        #     x *= 300
        # else:
        #     iterval = 50
        #     x = x[::iterval][:-1]
        #     #content = [np.mean(content[:i]) for i in range(len(content))]
        #     content = [np.mean(content[k*iterval:(k+1)*iterval]) for k in range(len(content)//iterval)]
        # if i == 2:
        #     plt.semilogy(x, content, marker='', ls='-', label=method_names[j], color=colors[j])
        # else:
        plt.plot(x, content, marker='', ls='-', label=method_names[j], color=colors[j])
    plt.xlabel('Iterations')
    plt.ylabel(plt_content[i])
    
    plt.grid()
    plt.legend()
    plt.savefig("visulizations/PACS/imgs/" + plt_names[i], bbox_inches='tight')
    plt.close()