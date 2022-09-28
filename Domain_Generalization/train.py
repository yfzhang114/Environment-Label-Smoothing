# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms, algorithms_cover
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default='/data1/User/datasets/DGdata/')
    parser.add_argument('--exp_name', type=str, default='dann_step1_baseline')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="DANN")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--com', action='store_true',
        help='use combination of classifiers')
    parser.add_argument('--cover', action='store_true',
        help='use cover for variance reduction')
    parser.add_argument('--cover_lambda', type=float, default=0.5,
             help='tradeoff weight for cover')
    parser.add_argument('--alpha_cover', type=float, default=0.0,
             help='alpha for cover')
    parser.add_argument('--w_erm', action='store_true',
            help='use combination of classifiers')
    parser.add_argument('--label_smooth', action='store_true',
            help='domain label smmoth')
    # parser.add_argument('--label_smooth', type=bool, default=False,
    #          help='domain label smmoth')
    parser.add_argument('--eps', type=float, default=0.1,
             help='eps for domain label smmoth')
    parser.add_argument('--gamma', type=float, default=1.0,
                help='use combination of classifiers')
    parser.add_argument('--steps', type=int, default=5000,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    hparams['com'] = args.com
    hparams['label_smooth'] = args.label_smooth
    hparams['eps'] = args.eps
    hparams['gamma'] = args.gamma
    hparams['alpha_cover'] = args.alpha_cover
    hparams['cover_lambda'] = args.cover_lambda
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    eval_batch = 64 #if args.algorithm != 'DRM' else 1
    train_loaders_eval = [FastDataLoader(
        dataset=env,
        batch_size=eval_batch,
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=eval_batch,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    if args.cover:
        algorithm_class = algorithms_cover.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
    else:
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams)
        if args.algorithm == 'ERMNP': 
            from domainbed.knn import MomentumQueue
            algorithm.eval_knn = MomentumQueue(algorithm.featurizer.n_outputs, sum([len(loader) for loader in train_loaders_eval]) * eval_batch, hparams['temperature'], hparams['k'], dataset.num_classes).cuda() # storing all features during test
            # algorithm.classifier = MomentumQueue(algorithm.featurizer.n_outputs, sum([len(loader) for loader in train_loaders_eval]) * eval_batch, hparams['temperature'], hparams['k'], dataset.num_classes).cuda() # storing all features during test

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    print("{} have {} paramerters in total".format(args.algorithm, sum(x.numel() for x in algorithm.parameters())))

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    best_valid = [0 for i in range(len(dataset))]
    best_test = [0 for i in range(len(dataset))]
    last_results_keys = None
    if dataset in ['OfficeHome', 'DomainNet']:
        n_steps *= 2
    losses_d, losses_e, gradc, gradcn, avg_acc_source, avg_acc_target = [], [], [], [], [], []

    if args.algorithm == 'ERMNP':
        bszs = 0
        while bszs * hparams['batch_size'] < hparams['queue_size']:
            minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
            algorithm.classifier.update_queue(algorithm.featurizer(torch.cat([x for x, y in minibatches_device])), torch.cat([y for x, y in minibatches_device]))
            bszs += 1

    # train_start_time = time.clock()
    for step in range(start_step, n_steps):
        # step_start_time = time.time()

        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]

        # if args.algorithm == 'ERMNP' and step % 10 == 0:
        #     for loader in train_loaders_eval:
        #         for x,y in loader:
        #             algorithm.classifier.update_queue(algorithm.featurizer(x.to(device)), y.to(device))

        step_vals = algorithm.update(minibatches_device, step)
        # checkpoint_vals['step_time'].append(time.time() - step_start_time)
        # print(step_vals['loss'])
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            require_ent = False
            avg_ps = []
            causals = []
            if args.algorithm == 'ERMNP':
                for loader in train_loaders_eval:
                    for x,y in loader:
                        algorithm.eval_knn.update_queue(algorithm.featurizer(x.to(device)), y.to(device))
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device, require_ent=require_ent, alg=args.algorithm)      
                results[name+'_acc'] = acc
                
            if require_ent:
                for i in range(len(avg_ps)//2):
                    for j in range(len(avg_ps)//2):
                        arr_i = avg_ps[i]; arr_j = avg_ps[j]
                        diff = torch.abs(arr_i-arr_j)
                        causals.append(torch.norm(diff,dim=0).item())
                    print(causals[i*len(avg_ps)//2:(i+1)*len(avg_ps)//2])
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })
            sum = 0
            for i in range(len(best_valid)):
                valid, test = float(results[results_keys[i*2]]), float(results[results_keys[i*2+1]])
                sum += test
                if valid > best_valid[i]:
                    best_valid[i] = valid
                    best_test[i] = test
            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")
            losses_e.extend(checkpoint_vals['gen_loss'])
            losses_d.extend(checkpoint_vals['grad_norm'])
            gradc.extend(checkpoint_vals['grad_center_norm'])
            gradcn.extend(checkpoint_vals['grad_cluster_param_new'])
            avg_acc_source.append(float(results[results_keys[args.test_envs[0]*2+1]]))
            avg_acc_target.append((sum-float(results[results_keys[args.test_envs[0]*2+1]]))/(len(eval_loaders)//2-1))
            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    # end_time = time.clock()
    save_checkpoint('model.pkl')
    print('best_valid',best_valid)
    print('best_test', best_test)
    # print('total time', end_time - train_start_time)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    save_dir = os.path.join('visulizations/PACS/dann_grad_dstep5_eps05/', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir + '/losses_e.npy', np.array(losses_e)) 
    np.save(save_dir + '/norm.npy', np.array(losses_d)) 
    np.save(save_dir + '/norm_c.npy', np.array(gradc)) 
    np.save(save_dir + '/norm_cn.npy', np.array(gradcn)) 
    np.save(save_dir + '/avg_acc_source.npy', np.array(avg_acc_source)) 
    np.save(save_dir + '/avg_acc_target.npy', np.array(avg_acc_target)) 