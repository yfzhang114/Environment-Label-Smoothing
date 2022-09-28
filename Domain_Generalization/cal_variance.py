import numpy as np
import PIL
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import  argparse

from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def variance_perclass(data_loaders, input_shape, num_cls, feature=False):
    if feature:
        network = torchvision.models.resnet50(pretrained=True)
        n_outputs = 2048
        del network.fc
        network.fc = Identity()
        centers = np.array([[np.zeros(n_outputs) for c in range(num_cls)] for l in range(len(data_loaders))])
        network.cuda()
        network.eval()
    else:
        centers = np.array([[np.zeros(input_shape) for c in range(num_cls)] for l in range(len(data_loaders))])
    
    variances_intra = np.array([[0. for c in range(num_cls)] for l in range(len(data_loaders))])
    variance_inter = np.array([0. for c in range(num_cls)])

    for k, loader in enumerate(data_loaders):
        xs = [[] for i in range(num_cls)]
        for i, (x, y) in enumerate(loader):
            x = network(x.cuda()).cpu().detach().numpy() if feature else x.numpy()
            xs[y].append(x)
        for i in range(num_cls):
            xs[i] = np.array(xs[i])
            centers[k,i] = np.mean(xs[i], axis=0)
            variances_intra[k,i] = np.std(xs[i], axis=0).mean()
    
    for i in range(num_cls):
        var = np.std(centers[:,i], axis=0).squeeze()
        variance_inter[i] = var.mean()
    print(variance_inter.mean(), np.mean(variances_intra))
    return variance_inter, np.mean(variances_intra, axis=0), variances_intra

def variance(data_loaders, input_shape, feature=False):

    if feature:
        network = torchvision.models.resnet50(pretrained=True)
        n_outputs = 2048
        del network.fc
        network.fc = Identity()
        centers = np.array([np.zeros(n_outputs) for l in range(len(data_loaders))])
        network.cuda()
        network.eval()
    else:
        centers = np.array([np.zeros(input_shape) for l in range(len(data_loaders))])
    variances_intra = np.array([0. for l in range(len(data_loaders))])
    variance_inter = 0

    for k, loader in enumerate(data_loaders):
        xs = np.array([np.zeros(input_shape) for l in range(len(loader))]) if not feature else np.array([np.zeros(n_outputs) for l in range(len(loader))])
        for i, (x, y) in enumerate(loader):
            xs[i] = x if not feature else network(x.cuda()).cpu().detach().numpy()
        centers[k] = np.mean(xs, axis=0)
        variances_intra[k] = np.std(xs, axis=0).mean()
    
    variance_inter = np.std(centers, axis=0).mean()
    return variance_inter, np.mean(variances_intra), variances_intra

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default='/data1/User/datasets/DGdata/')
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])

    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams('ERM', args.dataset)
    else:
        hparams = hparams_registry.random_hparams('ERM', args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy('file_system')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            [0], hparams)
    else:
        raise NotImplementedError
    input_shape = dataset.input_shape
    num_cls = dataset.num_classes

    in_splits = []

    train_loaders = [torch.utils.data.DataLoader(
            dataset.datasets[i],
            num_workers=1,
            batch_size=1) for i in range(len(dataset.datasets))]
    
    print(variance_perclass(train_loaders, input_shape, num_cls, feature=True))
    # print(variance(train_loaders, input_shape, feature=True))
