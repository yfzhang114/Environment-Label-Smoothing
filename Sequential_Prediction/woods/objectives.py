
"""Defining domain generalization algorithms"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import matplotlib.pyplot as plt

OBJECTIVES = [
    'ERM',
    'IRM',
    'VREx',
    'SD',
    'ANDMask',
    'IGA',
    'DANN',
    # 'Fish',
    # 'SANDMask',
    'IB_ERM_features',
    'IB_ERM_logits',
    'IB_IRM'
]

def cross_entropy_loss(pred_class_logits, gt_classes, eps=0.3, alpha=0.2, reduction='none'):
    num_classes = pred_class_logits.size(-1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_logits, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_logits, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    if reduction is not None:
        loss = loss.sum() / non_zero_cnt
    return loss


def get_objective_class(objective_name):
    """Return the objective class with the given name."""
    if objective_name not in globals():
        raise NotImplementedError("objective not found: {}".format(objective_name))
    return globals()[objective_name]


class Objective(nn.Module):
    """
    A subclass of Objective implements a domain generalization Gradients.
    Subclasses should implement the following:
    - update
    - predict
    """
    def __init__(self, hparams):
        super(Objective, self).__init__()

        self.hparams = hparams

    def backward(self, losses):
        """
        Computes the Gradients for model update

        Admits a list of unlabeled losses from the test domains: losses
        """
        raise NotImplementedError

class ERM(Objective):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(ERM, self).__init__(hparams)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out, features = self.model(all_x, ts)

        return out, features

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)

        # Get logit and make prediction on PRED_TIME
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, _ = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        labels_split = dataset.split_labels(all_y)

        # Compute loss for each environment
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx,:], labels_split[i,:,t_idx])

        # Compute objective
        objective = env_losses.mean()

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class DANN(ERM):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(DANN, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        self.register_buffer('update_count', torch.tensor([0]))

        num_domains = len(dataset.ENVS) - 1
        self.discriminator = MLP(self.model.state_size,
            num_domains, self.hparams)

        self.disc_opt = torch.optim.Adam(
            list(self.discriminator.parameters()),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.label_smooth = hparams['label_smooth']
        self.eps = hparams['eps']
        self.hparams = hparams

    def update(self, minibatches, dataset, device):
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches]).to(device)
        all_y = torch.cat([y for x, y in minibatches]).to(device)

        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, all_z = self.predict(all_x, ts, device)
        all_z = all_z.squeeze()
        disc_out = self.discriminator(all_z.squeeze())

        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])
        if self.label_smooth:
            disc_loss = cross_entropy_loss(disc_out, disc_labels, self.eps)
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(), [all_z], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
        else:
            # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
            out_split = dataset.split_output(out)
            labels_split = dataset.split_labels(all_y)

            env_losses = torch.zeros(out_split.shape[0]).to(device)
            for i in range(out_split.shape[0]):
                for t_idx in range(out_split.shape[2]):     # Number of time steps
                    env_losses[i] += self.loss_fn(out_split[i,:,t_idx,:], labels_split[i,:,t_idx])
            classifier_loss = env_losses.mean()
            objective = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()


class IRM(ERM):
    """
    Invariant Risk Minimization (IRM)
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(IRM, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']
        self.anneal_iters = self.hparams['anneal_iters']

        # Memory
        self.penalty = 0
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches_device, dataset, device):

        # Define stuff
        penalty_weight = (self.penalty_weight   if self.update_count >= self.anneal_iters 
                                                else 1.0)

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        # Get logit and make prediction on PRED_TIME
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, _ = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        labels_split = dataset.split_labels(all_y)

        # Compute loss and penalty for each environment 
        irm_penalty = torch.zeros(out_split.shape[0]).to(device)
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i,:,t_idx,:], labels_split[i,:,t_idx])
                irm_penalty[i] += self._irm_penalty(out_split[i,:,t_idx,:], labels_split[i,:,t_idx])

        # Compute objective
        irm_penalty = irm_penalty.mean()
        objective = env_losses.mean() + (penalty_weight * irm_penalty)

        # Reset Adam, because it doesn't like the sharp jump in gradient
        # magnitudes that happens at this step.
        if self.update_count == self.anneal_iters:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay'])

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Update memory
        self.update_count += 1

class VREx(ERM):
    """
    V-REx Objective from http://arxiv.org/abs/2003.00688
    """
    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(VREx, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']
        self.anneal_iters = self.hparams['anneal_iters']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches_device, dataset, device):

        # Define stuff
        penalty_weight = (self.penalty_weight   if self.update_count >= self.anneal_iters 
                                                else 1.0)

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        # Get logit and make prediction on PRED_TIME
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, _ = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        labels_split = dataset.split_labels(all_y)

        # Compute loss for each environment 
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

        # Compute objective
        mean = env_losses.mean()
        penalty = ((env_losses - mean) ** 2).mean()
        objective = mean + penalty_weight * penalty

        # Reset Adam, because it doesn't like the sharp jump in gradient
        # magnitudes that happens at this step.
        if self.update_count == self.anneal_iters:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay'])

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Update memory
        self.update_count += 1

class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(SD, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        # Get logit and make prediction on PRED_TIME
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, _ = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        labels_split = dataset.split_labels(all_y)

        # Compute loss for each environment 
        sd_penalty = torch.zeros(out_split.shape[0]).to(device)
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 
                sd_penalty[i] += (out_split[i, :, t_idx, :] ** 2).mean()

        sd_penalty = sd_penalty.mean()
        objective = env_losses.mean() + self.penalty_weight * sd_penalty

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(ANDMask, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.tau = self.hparams['tau']

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        # Get logit and make prediction on PRED_TIME
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, _ = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        labels_split = dataset.split_labels(all_y)

        # Compute loss for each environment 
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

        # Compute gradients for each env
        param_gradients = [[] for _ in self.model.parameters()]
        for env_loss in env_losses:

            env_grads = autograd.grad(env_loss, self.model.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)
            
        # Back propagate
        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.model.parameters())
        self.optimizer.step()

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(IGA, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        # Get logit and make prediction on PRED_TIME
        ts = torch.tensor(dataset.PRED_TIME).to(device)

        # There is an unimplemented feature of cudnn that makes it impossible to perform double backwards pass on the network
        # This is a workaround to make it work proposed by pytorch, but I'm not sure if it's the right way to do it
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        labels_split = dataset.split_labels(all_y)

        # Compute loss for each environment 
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

        # Get the gradients
        grads = []
        for env_loss in env_losses:

            # env_grad = autograd.grad(env_loss, self.model.parameters(), 
            #                             create_graph=True)
            env_grad = autograd.grad(env_loss, [p for p in self.model.parameters() if p.requires_grad], 
                                        create_graph=True)

            grads.append(env_grad)

        # Compute the mean loss and mean loss gradient
        mean_loss = env_losses.mean()
        mean_grad = autograd.grad(mean_loss, [p for p in self.model.parameters() if p.requires_grad], 
                                        create_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.penalty_weight * penalty_value

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        
# class Fish(ERM):
#     """
#     Implementation of Fish, as seen in Gradient Matching for Domain 
#     Generalization, Shi et al. 2021.
#     """

#     def __init__(self, model, dataset, loss_fn, optimizer, hparams):
#         super(Fish, self).__init__(model, dataset, loss_fn, optimizer, hparams)
        
#          # Hyper parameters
#         self.meta_lr = self.hparams['meta_lr']
        
#         self.model = model
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer

#     def create_copy(self, device):
#         self.model_inner = self.model
#         self.optimizer_inner = self.optimizer
#         self.loss_fn_inner = self.loss_fn 

#     def update(self, minibatches_device, dataset, device):
        
#         self.create_copy(minibatches_device[0][0].device)

#         ## Group all inputs and send to device
#         all_x = torch.cat([x for x,y in minibatches_device]).to(device)
#         all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
#         # Get logit and make prediction on PRED_TIME
#         ts = torch.tensor(dataset.PRED_TIME).to(device)


#         # There is an unimplemented feature of cudnn that makes it impossible to perform double backwards pass on the network
#         # This is a workaround to make it work proposed by pytorch, but I'm not sure if it's the right way to do it
#         with torch.backends.cudnn.flags(enabled=False):
#             out, _ = self.predict(all_x, ts, device)

#         # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
#         out_split = dataset.split_output(out)
#         labels_split = dataset.split_labels(all_y)

#         # Compute loss for each environment 
#         env_losses = torch.zeros(out_split.shape[0]).to(device)
#         for i in range(out_split.shape[0]):
#             for t_idx in range(out_split.shape[2]):     # Number of time steps
#                 env_losses[i] += self.loss_fn_inner(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

#         # Get the gradients
#         param_gradients = [[] for _ in self.model_inner.parameters()]
#         for env_loss in env_losses:
#             env_grads = autograd.grad(env_loss, self.model_inner.parameters(), retain_graph=True)
#             for grads, env_grad in zip(param_gradients, env_grads):
#                 grads.append(env_grad)
                
#       # Compute the meta penalty and update objective 
#         mean_loss = env_losses.mean()
#         meta_grad = autograd.grad(mean_loss, self.model.parameters(), 
#                                         create_graph=True)

#         # compute trace penalty
#         meta_penalty = 0
#         print(param_gradients[0])
#         for grad in param_gradients:
#             for g, meta_grad in zip(grad, meta_grad):
#                 meta_penalty += self.meta_lr * (g-meta_grad).sum() 

#         objective = mean_loss + meta_penalty

#         # Back propagate
#         self.optimizer.zero_grad()
#         objective.backward()
#         self.optimizer.step()
        
# class SANDMask(ERM):
#     """
#     Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
#     AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
#     """

#     def __init__(self, model, dataset, loss_fn, optimizer, hparams):
#         super(SANDMask, self).__init__(model, dataset, loss_fn, optimizer, hparams)

#         # Hyper parameters
#         self.tau = self.hparams['tau']
#         self.k = self.hparams['k']
#         self.betas = self.hparams['betas']

#         # Memory
#         self.register_buffer('update_count', torch.tensor([0]))

#     def mask_grads(self, tau, k, gradients, params, device):
#         '''
#         Mask are ranged in [0,1] to form a set of updates for each parameter based on the agreement 
#         of gradients coming from different environments.
#         '''
        
#         for param, grads in zip(params, gradients):
#             grads = torch.stack(grads, dim=0)
#             avg_grad = torch.mean(grads, dim=0)
#             grad_signs = torch.sign(grads)
#             gamma = torch.tensor(1.0).to(device)
#             grads_var = grads.var(dim=0)
#             grads_var[torch.isnan(grads_var)] = 1e-17
#             lam = (gamma * grads_var).pow(-1)
#             mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
#             mask = torch.max(mask, torch.zeros_like(mask))
#             mask[torch.isnan(mask)] = 1e-17
#             mask_t = (mask.sum() / mask.numel())
#             param.grad = mask * avg_grad
#             param.grad *= (1. / (1e-10 + mask_t))    

#     def update(self, minibatches_device, dataset, device):

#         ## Group all inputs and send to device
#         all_x = torch.cat([x for x,y in minibatches_device]).to(device)
#         all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
#         # Get logit and make prediction on PRED_TIME
#         ts = torch.tensor(dataset.PRED_TIME).to(device)
#         out, _ = self.predict(all_x, ts, device)

#         # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
#         out_split = dataset.split_output(out)
#         labels_split = dataset.split_labels(all_y)

#         # Compute loss for each environment 
#         env_losses = torch.zeros(out_split.shape[0]).to(device)
#         for i in range(out_split.shape[0]):
#             for t_idx in range(out_split.shape[2]):     # Number of time steps
#                 env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

#         # Compute the grads for each environment
#         param_gradients = [[] for _ in self.model.parameters()]
#         for env_loss in env_losses:

#             env_grads = autograd.grad(env_loss, self.model.parameters(), retain_graph=True)
#             for grads, env_grad in zip(param_gradients, env_grads):
#                 grads.append(env_grad)
            
#         # Back propagate with the masked gradients
#         self.optimizer.zero_grad()
#         self.mask_grads(self.tau, self.k, param_gradients, self.model.parameters(), device)
#         self.optimizer.step()

#         # Update memory
#         self.update_count += 1
                
class IB_ERM_features(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(IB_ERM_features, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.ib_weight = self.hparams['ib_weight']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        # Get time predictions and get logits
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, features = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        features_split = dataset.split_output(features)
        labels_split = dataset.split_labels(all_y)

        # For each environment, accumulate loss for all time steps
        ib_penalty = torch.zeros(out_split.shape[0]).to(device)
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                # Compute the penalty
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 
                # Compute the information bottleneck
                ib_penalty[i] += features_split[i, :, t_idx, :].var(dim=0).mean()

        objective = env_losses.mean() + (self.ib_weight * ib_penalty.mean())

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Update memory
        self.update_count += 1
         
class IB_ERM_logits(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(IB_ERM_logits, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.ib_weight = self.hparams['ib_weight']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        # Get time predictions and get logits
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, features = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        features_split = dataset.split_output(features)
        labels_split = dataset.split_labels(all_y)

        # For each environment, accumulate loss for all time steps
        ib_penalty = torch.zeros(out_split.shape[0]).to(device)
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                # Compute the penalty
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 
                # Compute the information bottleneck
                ib_penalty[i] += out_split[i, :, t_idx, :].var(dim=0).mean()

        objective = env_losses.mean() + (self.ib_weight * ib_penalty.mean())

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Update memory
        self.update_count += 1

class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(IB_IRM, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.ib_weight = self.hparams['ib_weight']
        self.ib_anneal = self.hparams['ib_anneal']
        self.irm_weight = self.hparams['irm_weight']
        self.irm_anneal = self.hparams['irm_anneal']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches_device, dataset, device):

        # Get penalty weight
        ib_penalty_weight = (self.ib_weight if self.update_count
                          >= self.ib_anneal else
                          0.0)
        irm_penalty_weight = (self.irm_weight if self.update_count
                          >= self.irm_anneal else
                          1.0)

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        # Get time predictions and get logits
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out, features = self.predict(all_x, ts, device)

        # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
        out_split = dataset.split_output(out)
        features_split = dataset.split_output(features)
        labels_split = dataset.split_labels(all_y)

        # For each environment, accumulate loss for all time steps
        ib_penalty = torch.zeros(out_split.shape[0]).to(device)
        irm_penalty = torch.zeros(out_split.shape[0]).to(device)
        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                # Compute the penalty
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 
                # Compute the information bottleneck penalty
                ib_penalty[i] += features_split[i, :, t_idx, :].var(dim=0).mean()
                # Compute the invariant risk minimization penalty
                irm_penalty[i] += self._irm_penalty(out_split[i,:,t_idx,:], labels_split[i,:,t_idx])

        objective = env_losses.mean() + ib_penalty_weight * ib_penalty.mean() + irm_penalty_weight * irm_penalty.mean()

        if self.update_count == self.ib_anneal or self.update_count == self.irm_anneal:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay'])

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Update memory
        self.update_count += 1