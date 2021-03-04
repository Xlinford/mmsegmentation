import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torchvision import transforms
import ipdb

from ..builder import LOSSES
from .utils import weight_reduce_loss
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.ops import corner_crop


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class PixelwiseContrastiveLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=0.1):
        super(PixelwiseContrastiveLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.temp = 10
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def feature_prepare(self, seg_logit, seg_label, img_metas):
        """Crop features

        Crop features like [batch, channel, cropsize, cropsize] witch have overlapping region into overlapping size[y, x]
        Args:
            seg_logit: list of tensor, features from encoder [img1:[B, C, H, W], img2:[B, C, H, W]], [H, W] is crop size
            seg_label: tensor, logit from last layer [B, C, H, W], C is class number
            img_metas: dict, from data enhancement process record

        Returns:
            seg_logit: list, length = Batch, size like [C, y, x]
            seg_label: list, length = Batch, size like [C, y, x]
        """

        feat = []
        pseudo_labels = []
        feat1 = [i for i in range(len(img_metas))]
        pseudo_labels1 = feat1.copy()

        for i in range(len(img_metas)):
            crop_region = img_metas[i]['cover_crop_box']
            # feat.append(seg_logit[0][i, :, :, :])
            feat.append([logit[i, :, :, :] for logit in seg_logit])
            pseudo_labels.append([label[i, :, :, :] for label in seg_label])
            # pseudo_labels.append(seg_label[i, :, :, :])
            feat1[i], pseudo_labels1[i] = corner_crop(crop_region, feat[i], pseudo_labels[i])

        # seg_logit = feat
        # seg_label = pseudo_labels
        return feat1, pseudo_labels1

    def calc_neg_logits(self, feats, pseudo_labels, neg_feats, neg_pseudo_labels):
        """calculate negative feature pair

        Args:
            feats: [128,h*w]
            pseudo_labels: [1,h*w]
            neg_feats: [128,b]
            neg_pseudo_labels: [1,b]
            temp: 10

        Returns:

        """

        pseudo_labels1 = pseudo_labels.permute(1, 0)  # [h*w,1]
        # neg_pseudo_labels = neg_pseudo_labels.unsqueeze(0)  # [1,b]
        # negative sampling mask (Nxb)
        neg_mask = (pseudo_labels1 != neg_pseudo_labels).float()
        neg_scores = (feats.T @ neg_feats) / 100  # negative scores (Nxb)
        return (neg_mask.float() * torch.exp(neg_scores)).sum(-1)

    def forward(self,
                feats,
                pseudo_logits,
                cls_score,
                gt_semantic_seg,
                img_metas,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        # calculate the negative logits of proposed loss function

        # feats1: features of the overlapping region in the first crop (NxC)....
        # feats2: features of the overlapping region in the second crop (NxC)....
        # neg_feats: all selected negative features (nxC)....
        # pseudo_labels1: pseudo labels for feats1 (N)
        # pseudo_logits1: confidence for feats1 (N)....
        # pseudo_logits2: confidence for feats2 (N)....
        # neg_pseudo_labels: pseudo labels for neg_feats (n)
        # gamma: the threshold value for positive filtering
        # temp: the temperature value
        # b: an integer to divide the loss computation into several parts
        # N: overlapping region;    n: crop region

        b = 300
        gamma = 0.75
        n = img_metas[1]['img_shape']
        n = n[0] * n[1]
        loss = []
        pos_feats, pos_pseudo_labels = self.feature_prepare(feats, pseudo_logits, img_metas)

        for j in range(len(pos_feats)):
            feats1 = torch.reshape(pos_feats[j][0], (128, -1))
            feats2 = torch.reshape(pos_feats[j][1], (128, -1))
            neg_feats = torch.reshape(feats[0][j, :, :, :], (128, -1))  # change dim
            pseudo_logits1 = pos_pseudo_labels[j][0]
            pseudo_logits2 = pos_pseudo_labels[j][1]
            pseudo_labels1 = torch.reshape(torch.argmax(pseudo_logits1, dim=0), (1, -1))
            # pseudo_logits-->[B,C,H,W] pos_pseudo_labels1-->[1,H,W]
            neg_pseudo_labels1 = torch.argmax(torch.squeeze(pseudo_logits[0][j], dim=0), dim=0)
            neg_pseudo_labels1 = torch.reshape(neg_pseudo_labels1, (1, -1))

            # print_log(f'input{j}{[feats1.size(), feats2.size()]}', logger=get_root_logger())
            pos1 = (feats1 * feats2.detach()).sum(0) / self.temp  # positive scores (N)
            # try:
            #     pos1 = (feats1 * feats2.detach()).sum(0) / temp  # positive scores (N)
            # except:
            #     import ipdb
            #     ipdb.set_trace()
            neg_logits = torch.zeros(pos1.size(0), device=pos1.device)  # initialize negative scores (n)N
            # divide the negative logits computation into several parts
            # in each part, only b negative samples are considered

            for i in range((n - 1) // b + 1):
                neg_feats_i = neg_feats[:, i * b:(i + 1) * b]
                neg_pseudo_labels_i = neg_pseudo_labels1[:, i * b:(i + 1) * b]
                neg_logits_i = torch.utils.checkpoint.checkpoint(
                    self.calc_neg_logits,
                    feats1,  # [128,h*w]
                    pseudo_labels1,  # [1,h*w]
                    neg_feats_i,  # [128,b]
                    neg_pseudo_labels_i)  # [1,b]
                neg_logits += neg_logits_i
            # compute the loss for the first crop

            logits1 = torch.exp(pos1) / (torch.exp(pos1) + neg_logits + 1e-8)
            lossn = -torch.log(logits1 + 1e-8)  # (N)
            dir_mask1 = (pseudo_logits1 < pseudo_logits2)  # directional mask (N)
            pos_mask1 = (pseudo_logits2 > gamma)  # positive filtering mask (N)
            mask1 = torch.reshape(torch.argmax((dir_mask1 * pos_mask1).float(), dim=0), (1, -1)).squeeze(0)
            # final loss for the first crop

            loss.append((mask1 * lossn).sum() / (mask1.sum() + 1e-8))

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.cls_criterion(
            cls_score,
            gt_semantic_seg,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        loss1 = sum(loss)
        loss2 = self.loss_weight * loss1 + loss_cls

        return loss2



