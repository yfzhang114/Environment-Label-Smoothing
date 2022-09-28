"""
Rotating MNIST dataset
Domain: anti-clockwise rotating angle (theta)
Source domains: theta in (0, 45 degree)
    Amount: 60000
Target domains: theta in (45, 360 degree)
    Amount: 60000 * 7
"""

"""
Download data if needed
"""
from model import download
download()

"""
Visualize the data
"""
import matplotlib.pyplot as plt
from model import RotateMNIST, ContinousRotateMNIST

for i in range(8):
    dataset = RotateMNIST(rotate_angle=(i*45,i*45+45))
    if i == 0:
        dname = 'Source'
    else:
        dname = f'Sub Target #{i}'
    print(dname)
    fig, ax = plt.subplots(1, 10, figsize=(18,1.5))
    for j in range(10):
        img, label, angle, _ = dataset[j]
        angle = angle[0] * 360
        ax[j].imshow(img[0])
        ax[j].set_title(f'Label: {label}\nRot: {angle:.0f}')
    plt.show()
    plt.close()

from easydict import EasyDict
from model import set_default_args, print_args
from model import SO, ADDA, DANN, CUA, CIDA, PCIDA
from torch.utils.data import DataLoader

opt = EasyDict()
# choose the method from ["CIDA", "PCIDA", "SO", "ADDA", "DANN" "CUA"]
opt.model = "DANN"
# choose run on which device ["cuda", "cpu"]
opt.device = "cuda"
set_default_args(opt)
print_args(opt)
# build dataset and data loader
dataset = RotateMNIST(rotate_angle=(0, 360))
train_dataloader = DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
test_dataloader = DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
# build model
model_pool = {
    'SO': SO,
    'CIDA': CIDA,
    'PCIDA': PCIDA,
    'ADDA': ADDA,
    'DANN': DANN,
    'CUA': CUA,
}
model = model_pool[opt.model](opt)
model = model.to(opt.device)

"""
Training the model from the scratch
"""
best_acc_target = 0
if not opt.continual_da:
    # Single Step Domain Adaptation
    for epoch in range(opt.num_epoch):
        model.learn(epoch, train_dataloader)
        if (epoch + 1) % 10 == 0:
            acc_target = model.eval_mnist(test_dataloader)
            if acc_target > best_acc_target:
                print('Best acc target. saved.')
                model.save()
else:
    # continual DA training
    continual_dataset = ContinousRotateMNIST()

    print('===> pretrain the classifer')
    model.prepare_trainer(init=True)
    for epoch in range(opt.num_epoch_pre):
        model.learn(epoch, train_dataloader, init=True)
        if (epoch + 1) % 10 == 0:
            model.eval_mnist(test_dataloader)
    print('===> start continual DA')
    model.prepare_trainer(init=False)
    for phase in range(opt.num_da_step):
        continual_dataset.set_phase(phase)
        print(f'Phase {phase}/{opt.num_da_step}')
        print(f'#source {len(continual_dataset.ds_source)} #target {len(continual_dataset.ds_target[phase])} #replay {len(continual_dataset.ds_replay)}')
        continual_dataloader = DataLoader(
            dataset=continual_dataset,
            shuffle=True,
            batch_size=opt.batch_size,
            num_workers=4,
        )
        for epoch in range(opt.num_epoch_sub):
            model.learn(epoch, continual_dataloader, init=False)
            if (epoch + 1) % 10 == 0:
                model.eval_mnist(test_dataloader)

        target_dataloader = DataLoader(
            dataset=continual_dataset.ds_target[phase],
            shuffle=True,
            batch_size=opt.batch_size,
            num_workers=4,
        )
        acc_target = model.eval_mnist(test_dataloader)
        if acc_target > best_acc_target:
            print('Best acc target. saved.')
            model.save()
        data_tuple = model.gen_data_tuple(target_dataloader)
        continual_dataset.ds_replay.update(data_tuple)  
"""
Load the pretrained model
"""
model.load()
model.gen_result_table(test_dataloader)
