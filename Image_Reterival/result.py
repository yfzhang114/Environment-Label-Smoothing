mean = [[0.12923385128378867, 0.32917848527431487, 4.2074590682983395, 5.381637763977051,], 
[0.27424232065677645, 0.5235896348953247, 2.231031763553619, 2.9850695967674254, ],
[0.17577361166477204,0.4030865550041199, 4.365029335021973, 4.736916542053223,],
[0.24205589294433594, 0.4709254026412964, 2.1547161459922792,2.837662672996521, ],
[0.14995907470583916,0.4373594269156456,2.2582317352294923,2.312002789974213,],
[0.19678926095366478, 0.44505256712436675, 1.9305160164833068, 1.8429392218589782,],
[0.18515243381261826,0.6943907335400581,1.9229639172554016, 2.1786173462867735,],
[0.13651328459382056, 0.7083251520991325, 2.128530776500702, 1.7286038041114806,],
[0.11500186789780856, 0.494458919018507, 1.4342263877391814, 1.1562542200088501],
[0.14817044287919998, 0.427288855612278,1.9765880465507508,  1.5862125873565673,],
[0.6544407922774553, 2.8891056776046753, 1.2569275379180909, 1.5372634321451186,],
[0.27190949991345403, 3.201848286390304, 1.8299665302038193, 1.8178970694541932,],
[0.5770932335406542, 2.741451719403267, 1.4664539307355882, 1.5666474163532258,],
[0.2960498809814453, 1.351430407166481, 1.872667282819748, 2.286483755707741],
[0.2852806646376848, 2.8819854974746706, 1.2971589416265488, 1.0912617415189743,],
[1.0535070091485976, 2.444656363129616,1.5353894472122191,1.5082878351211548,],
[0.6750239424407483, 1.3086718782782554,1.6395948648452758, 1.385527667403221, ],
[1.013559013977647, 2.811639553308487, 2.0858230590820312, 1.9566635251045228, ],
[0.6685243174433708, 3.395494520664215, 1.5754039019346238, 1.4354386270046233,]
]
std = [[0.03866294694513128, 0.11080794443999041, 0.8060715059189711, 0.6329459880646423], 
[.25437298725917906, 0.2315479957501956, 0.7678407693384962,  1.0125494482097628],
[0.0661852482187546, 0.17585667219966628,  0.7873653623625182, 1.300903155530249],
[0.11488171432310426,0.345442189827558, 0.6716067683949429, 0.9408420094706241 ],
[0.060970126809802114, 0.4365388200329665, 1.1860514421284494,0.7784503037385071],
[0.10892157687831773,0.2975852369391704, 0.8088439552230986, 1.1002981686076025],
[0.09883639504268163, 0.6867902206728431,0.7202608467462448,1.00655814659257],
[0.07920770554727505, 0.429090361653401, 1.0746226631841374,0.8230243543572741 ],
[0.16911643042868732, 0.27547977698209897, 0.5817896564506787, 0.2515940481320079],
[0.10238149117583237, 0.16695442504661692, 1.0039056597287834, 0.372619908536996],
[0.922900895110043, 1.3914254544881186, 0.6329184776349063, 0.9812624899843104],
[0.09382022408364837, 2.6823554964640492, 1.3127490282347738, 1.1716010225965319],
[0.6099640317945947, 3.421268924652168, 0.8563086042745283, 0.7469301218965083],
[0.13342261358350999, 0.9589139072386343, 0.9924667652898592, 0.9454310118764402], 
[0.23359198690025373, 1.6605297512094332, 0.5637474795891855, 0.46832530390273946],
[ 1.585783724941088, 1.3337967595688969, 1.285201474939728, 0.9366384494538765],
[0.6572400972448255, 0.9690815422987749, 1.0853398025065573, 0.9963291875754547],
[1.1646479277002773, 1.662434801577887,1.092434783572759,  1.0606974056505785 ],
[1.481716857275175, 1.9690268019765003, 0.6452157931067393,0.8400998244465404]
]

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
sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=1.8)
iters = np.array([0, 5, 10, 20, 50, 100, 0, 5, 10, 20, 50, 100, 0, 5, 10, 20, 50, 100]).reshape((-1,1))
models = np.array(['DANN','DANN','DANN','DANN','DANN','DANN','DANS','DANS','DANS','DANS','DANS','DANS','ERM','ERM','ERM','ERM','ERM','ERM']).reshape((-1,1))
dro = [0, 8.08, 3.12, 0, 0, 0]
unitdro = [0, 54.25, 62.83, 67.43, 72.0, 71.6]
resnet = [0, 42.85, 55.37, 58.66, 65.83, 64.98]
values = np.array([0, 8.08, 3.12, 0, 0, 0, 0, 54.25, 62.83, 67.43, 72.0, 71.6, 0, 42.85, 55.37, 58.66, 65.83, 64.98]).reshape((-1,1))
data = np.concatenate((iters,models,values), axis=-1)
df = pd.DataFrame(data)
df.columns = ["Iterations", "Models", "Average mAP"]
df["Iterations"] = pd.to_numeric(df['Iterations'])
df["Average mAP"] = pd.to_numeric(df['Average mAP'])
print(df)
sns.lineplot(data=df, x="Iterations", y="Average mAP", hue="Models", palette="Set2", style="Models", linewidth=3.)
plt.tight_layout(h_pad=-0.1)
pdf = PdfPages('dann_map.pdf')
pdf.savefig()
plt.close()
pdf.close()

sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=1.8)
iters = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]).reshape((-1,1))
models = np.array(['DANN','DANN','DANN','DANN','DANN','DANN','DANS','DANS','DANS','DANS','DANS','DANS']).reshape((-1,1))
dro = [0, 9.789 , 27.457, 17.982, 11.080, 19.111]
unitdro = [0, 0.155, 0.156, 0.153, 0.159, 0.157]
values = np.array([0, 9.789 , 27.457, 17.982, 11.080, 19.111, 0, 0.155, 0.156, 0.153, 0.159, 0.157]).reshape((-1,1))
data = np.concatenate((iters,models,values), axis=-1)
df = pd.DataFrame(data)
df.columns = ["Iterations", "Models", "Discrimination Loss"]
df["Iterations"] = pd.to_numeric(df['Iterations'])
df["Discrimination Loss"] = pd.to_numeric(df['Discrimination Loss'])
print(df)
sns.lineplot(data=df, x="Iterations", y="Discrimination Loss", hue="Models", palette="Set2", style="Models", linewidth=3.)
plt.tight_layout(h_pad=-0.1)
pdf = PdfPages('dann_loss.pdf')
pdf.savefig()
plt.close()
pdf.close()

sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=1.8)
iters = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]).reshape((-1,1))
models = np.array(['DANN','DANN','DANN','DANN','DANN','DANN','DANS','DANS','DANS','DANS','DANS','DANS']).reshape((-1,1))
unitdro = [0, 9.928 , 8.422, 5.433, 2.670, 1.727]
dro = [0, 9.981, 22.933,  18.397, 34.5469, 17.983]
resnet = [0, 9.921, 8.146, 5.344,2.859, 1.781]
values = np.array([0, 9.928 , 8.422, 5.433, 2.670, 1.727, 0, 9.981, 22.933,  18.397, 34.5469, 17.983, 0, 9.921, 8.146, 5.344,2.859, 1.781]).reshape((-1,1))
data = np.concatenate((iters,models,values), axis=-1)
df = pd.DataFrame(data)
df.columns = ["Iterations", "Models", "Classification Loss"]
df["Iterations"] = pd.to_numeric(df['Iterations'])
df["Classification Loss"] = pd.to_numeric(df['Classification Loss'])
print(df)
sns.lineplot(data=df, x="Iterations", y="Classification Loss", hue="Models", palette="Set2", style="Models", linewidth=3.)
plt.tight_layout(h_pad=-0.1)
pdf = PdfPages('dann_cls.pdf')
pdf.savefig()
plt.close()
pdf.close()