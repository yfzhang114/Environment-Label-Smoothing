# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import logging

import torch.nn.functional as F
from torch import nn

from fastreid.modeling.ops import meta_linear, meta_norm
from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
from torch.autograd import Function

# logger = logging.getLogger(__name__)

@REID_HEADS_REGISTRY.register()
class MetalearningHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        norm_opt = dict()
        norm_opt['BN_AFFINE'] = cfg.MODEL.NORM.BN_AFFINE
        norm_opt['BN_RUNNING'] = cfg.MODEL.NORM.BN_RUNNING
        norm_opt['IN_AFFINE'] = cfg.MODEL.NORM.IN_AFFINE
        norm_opt['IN_RUNNING'] = cfg.MODEL.NORM.IN_RUNNING

        norm_opt['BN_W_FREEZE'] = cfg.MODEL.NORM.BN_W_FREEZE
        norm_opt['BN_B_FREEZE'] = cfg.MODEL.NORM.BN_B_FREEZE
        norm_opt['IN_W_FREEZE'] = cfg.MODEL.NORM.IN_W_FREEZE
        norm_opt['IN_B_FREEZE'] = cfg.MODEL.NORM.IN_B_FREEZE

        norm_opt['BIN_INIT'] = cfg.MODEL.NORM.BIN_INIT
        norm_opt['IN_FC_MULTIPLY'] = cfg.MODEL.NORM.IN_FC_MULTIPLY

        # Pooling
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.pool_layer = Flatten()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        # Bottleneck
        self.BOTTLENECK_flag = cfg.META.BOTTLENECK.DO_IT # True
        if self.BOTTLENECK_flag:
            bottleneck_in_feat = cfg.MODEL.HEADS.IN_FEAT
            bottleneck_out_feat = cfg.META.BOTTLENECK.REDUCTION_DIM
            self.bottleneck = bottleneck_layer(bottleneck_in_feat, bottleneck_out_feat, cfg, norm_opt)
            in_feat = bottleneck_out_feat
        else:
            in_feat = cfg.MODEL.HEADS.IN_FEAT

        # BNNeck
        # self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.classifier_norm = meta_norm(cfg.MODEL.NORM.TYPE_CLASSIFIER, in_feat, norm_opt=norm_opt, bias_freeze=True)
        # self.classifier_norm = meta_norm(cfg.MODEL.NORM.TYPE_CLASSIFIER, in_feat, norm_opt=norm_opt)

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        num_domains = cfg.META.DATA.NUM_DOMAINS
        if cls_type == 'linear':          self.classifier_fc = meta_linear(in_feat, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier_fc = ArcSoftmax(cfg, in_feat, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier_fc = CircleSoftmax(cfg, in_feat, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier_fc = AMSoftmax(cfg, in_feat, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax' and 'circleSoftmax'.")
        
        self.classifier_d = GRL_classifier(in_feat, num_domains)
        self.classifier_d.apply(weights_init_classifier)
        self.classifier_fc.apply(weights_init_classifier)

    def forward(self, features, targets=None, opt = None, cls=False):

        if self.BOTTLENECK_flag:
            global_feat = self.pool_layer(features) # 4D -> 4D[...,1,1]
            bottleneck_feat = self.bottleneck(global_feat[...,0,0], opt)
            bn_feat = self.classifier_norm(bottleneck_feat, opt)
        else:
            global_feat = self.pool_layer(features) # 4D -> 4D[...,1,1]
            bn_feat = self.classifier_norm(global_feat, opt)
        if len(bn_feat.shape) == 4:
            bn_feat = bn_feat[..., 0, 0]

        if not self.training and not cls: return bn_feat # Evaluation

        if self.classifier_fc.__class__.__name__ in ['Linear', 'meta_linear']:
            cls_outputs = self.classifier_fc(bn_feat, opt)
            d_cls_outputs = self.classifier_d(bn_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier_fc.weight) # compute accuracy
        else:
            cls_outputs = self.classifier_fc(bn_feat, targets)
            d_cls_outputs = self.classifier_d(bn_feat)
            pred_class_logits = self.classifier_fc.s * \
                                F.linear(F.normalize(bn_feat), F.normalize(self.classifier_fc.weight))

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "domain_logits": d_cls_outputs,
            "pooled_features": global_feat[..., 0, 0],
            "bn_features": bn_feat,
        }

class bottleneck_layer(nn.Module):

    def __init__(self, in_feat, reduction_dim, cfg, norm_opt):
        super(bottleneck_layer, self).__init__()

        self.bn_flag = cfg.META.BOTTLENECK.NORM
        # NORM_SPLIT = cfg.MODEL.HEADS.NORM_SPLIT

        self.fc = meta_linear(in_feat, reduction_dim, bias=False)
        self.fc.apply(weights_init_kaiming)
        # self.fc = nn.Linear(in_feat, reduction_dim, bias=False)

        if self.bn_flag:
            self.norm = meta_norm(cfg.MODEL.NORM.TYPE_BOTTLENECK, reduction_dim, norm_opt=norm_opt, bias_freeze=True)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.lrelu.apply(weights_init_kaiming)

    def forward(self, x, opt = None):

        x = self.fc(x, opt)

        while len(x.shape) < 4:
            x = x.unsqueeze(-1) # 2dim -> 4dim
        if self.bn_flag:
            x = self.norm(x, opt)
            # x = x[...,0,0] # 4dim -> 2dim

        x = self.lrelu(x)

        return x

        # Define the ResNet50-based Model
class GRL_classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GRL_classifier, self).__init__()

        # fc(1024)->bn->lrelu->dp->fc(512)->bn->lrelu->dp->fc(8)
        self.domain_classifier = nn.Linear(input_dim, output_dim, bias=True)


    def forward(self, f, alpha=1.0):
        reverse_f = ReverseLayerF.apply(f, alpha) # [128, 800]
        domain_output = self.domain_classifier(reverse_f)

        return domain_output
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None