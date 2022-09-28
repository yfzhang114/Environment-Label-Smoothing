"""Defining the training functions that are used to train and evaluate models"""

import time
import numpy as np

import torch
from torch import nn, optim

from woods import datasets
from woods import models
from woods import objectives
from woods import hyperparams
from woods import utils

## Train function
def train_step(model, objective, dataset, in_loaders_iter, device):
    """ Train a single training step for a model

    Args:
        model: nn model defined in a models.py
        objective: objective we are using for training
        dataset: dataset object we are training on
        in_loaders_iter: iterable of iterable of data loaders
        device: device on which we are training
    """
    model.train()

    ts = torch.tensor(dataset.PRED_TIME).to(device)
        

    # Get batch and Send everything in an array
    batch_loaders = next(in_loaders_iter)
    minibatches_device = [(x, y) for x,y in batch_loaders]

    objective.update(minibatches_device, dataset, device)

    return model

def train(flags, training_hparams, model, objective, dataset, device):
    """ Train a model on a given dataset with a given objective

    Args:
        flags: flags from argparse
        training_hparams: training hyperparameters
        model: nn model defined in a models.py
        objective: objective we are using for training
        dataset: dataset object we are training on
        device: device on which we are training
    """
    record = {}
    step_times = []
    
    t = utils.setup_pretty_table(flags)

    train_names, train_loaders = dataset.get_train_loaders()
    n_batches = np.sum([len(train_l) for train_l in train_loaders])
    train_loaders_iter = zip(*train_loaders)

    for step in range(1, dataset.N_STEPS + 1):

        ## Make training step and report accuracies and losses
        step_start = time.time()
        model = train_step(model, objective, dataset, train_loaders_iter, device)
        step_times.append(time.time() - step_start)

        if step % dataset.CHECKPOINT_FREQ == 0 or (step-1)==0:

            val_start = time.time()
            checkpoint_record = get_accuracies(objective, dataset, device)
            val_time = time.time() - val_start

            record[str(step)] = checkpoint_record

            if dataset.TASK == 'regression':
                t.add_row([step] 
                        + ["{:.1e} :: {:.1e}".format(record[str(step)][str(e)+'_in_loss'], record[str(step)][str(e)+'_out_loss']) for e in dataset.ENVS] 
                        + ["{:.1e}".format(np.average([record[str(step)][str(e)+'_loss'] for e in train_names]))] 
                        + ["{:.2f}".format((step*len(train_loaders)) / n_batches)]
                        + ["{:.2f}".format(np.mean(step_times))] 
                        + ["{:.2f}".format(val_time)])
            else:
                t.add_row([step] 
                        + ["{:.2f} :: {:.2f}".format(record[str(step)][str(e)+'_in_acc'], record[str(step)][str(e)+'_out_acc']) for e in dataset.ENVS] 
                        + ["{:.2f}".format(np.average([record[str(step)][str(e)+'_loss'] for e in train_names]))] 
                        + ["{:.2f}".format((step*len(train_loaders)) / n_batches)]
                        + ["{:.2f}".format(np.mean(step_times))] 
                        + ["{:.2f}".format(val_time)])

            step_times = [] 
            print("\n".join(t.get_string().splitlines()[-2:-1]))

    return model, record, t

def get_accuracies(objective, dataset, device):
    """ Get accuracies for all splits using fast loaders

    Args:
        objective: objective we are using for training
        dataset: dataset object we are training on
        device: device on which we are training
    """

    # Get loaders and their names
    val_names, val_loaders = dataset.get_val_loaders()

    ## Get test accuracy and loss
    record = {}
    for name, loader in zip(val_names, val_loaders):

        if dataset.SETUP == 'seq':
            accuracy, loss = get_split_accuracy_seq(objective, dataset, loader, device)
        
            record.update({ name+'_acc': accuracy,
                            name+'_loss': loss})

        elif dataset.SETUP == 'step':
            accuracies, losses = get_split_accuracy_step(objective, dataset, loader, device)

            for i, e in enumerate(name):
                record.update({ e+'_acc': accuracies[i],
                                e+'_loss': losses[i]})

    return record

def get_split_accuracy_seq(objective, dataset, loader, device):
    """ Get accuracy and loss for a dataset that is of the `seq` setup

    Args:
        objective: objective we are using for training
        dataset: dataset object we are training on
        loader: data loader of which we want the accuracy
        device: device on which we are training
    """

    ts = torch.tensor(dataset.PRED_TIME).to(device)

    losses = 0
    nb_correct = 0
    nb_item = 0
    with torch.no_grad():

        for b, (data, target) in enumerate(loader):

            data, target = data.to(device), target.to(device)

            all_out, _ = objective.predict(data, ts, device)

            for i, t in enumerate(ts):
                losses += objective.loss_fn(all_out[:,i,...], target[:,i])

            # get train accuracy and save it
            pred = all_out.argmax(dim=2)
            nb_correct += pred.eq(target).cpu().sum()
            nb_item += target.numel()

        show_value = nb_correct.item() / nb_item

    return show_value, losses.item() / len(loader)

def get_split_accuracy_step(objective, dataset, loader, device):
    """ Get accuracy and loss for a dataset that is of the `step` setup

    Args:
        objective: objective we are using for training
        dataset: dataset object we are training on
        loader: data loader of which we want the accuracy
        device: device on which we are training
    """

    ts = torch.tensor(dataset.PRED_TIME).to(device)

    losses = torch.zeros(*ts.shape).to(device)
    nb_correct = torch.zeros(*ts.shape).to(device)
    nb_item = torch.zeros(*ts.shape).to(device)
    with torch.no_grad():

        for b, (data, target) in enumerate(loader):

            data, target = data.to(device), target.to(device)

            all_out, _ = objective.predict(data, ts, device)

            for i, t in enumerate(ts):
                losses[i] += objective.loss_fn(all_out[:,i,...], target[:,i])

            pred = all_out.argmax(dim=2)
            nb_correct += torch.sum(pred.eq(target), dim=0)
            nb_item += pred.shape[0]
            
    return (nb_correct / nb_item).tolist(), (losses/len(loader)).tolist()