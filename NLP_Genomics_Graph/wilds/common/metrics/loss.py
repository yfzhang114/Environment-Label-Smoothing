import torch
from wilds.common.utils import avg_over_groups, maximum
from wilds.common.metrics.metric import ElementwiseMetric, Metric, MultiTaskMetric
import torch.nn.functional as F

class Loss(Metric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        return self.loss_fn(y_pred, y_true)

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)

class ElementwiseLoss(ElementwiseMetric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        return self.loss_fn(y_pred, y_true)

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)

class CrossEntropyLabelSmooth(ElementwiseMetric):
    def __init__(self, loss_fn, eps, name=None):
        self.loss_fn = loss_fn
        self.eps = eps
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true, alpha=0.2, reduction='none'):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        num_classes = y_pred.size(1)

        if self.eps >= 0:
            smooth_param = self.eps
        else:
            # Adaptive label smooth regularization
            soft_label = F.softmax(y_pred, dim=1)
            smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), y_true].unsqueeze(1)

        log_probs = F.log_softmax(y_pred, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (num_classes - 1)
            targets.scatter_(1, y_true.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).sum(dim=1)

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
        if reduction is not None:
            loss = loss.sum() / non_zero_cnt

        return loss

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)

class MultiTaskLoss(MultiTaskMetric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn # should be elementwise
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            flattened_y_pred = flattened_y_pred.float()
            flattened_y_true = flattened_y_true.float()
        elif isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
            flattened_y_true = flattened_y_true.long()
        flattened_loss = self.loss_fn(flattened_y_pred, flattened_y_true)
        return flattened_loss

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)
