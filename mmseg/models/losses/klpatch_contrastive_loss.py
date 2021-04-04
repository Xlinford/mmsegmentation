import torch
import mmcv
import torch.nn as nn
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torchvision import transforms
import ipdb
from time import time
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
                  ignore_index=-100,
                  **kwargs):
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
                         ignore_index=255,
                         **kwargs):
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
                       ignore_index=None,
                       **kwargs):
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
class KLPatchContrastiveLossBatch(nn.Module):
    """KLPatchContrastiveLoss contained CrossEntropyLoss.

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
                 loss_weight=0.1,
                 patch_size=16, ):
        super(KLPatchContrastiveLossBatch, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.patch_size = patch_size
        self.temp = 50
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def cross_class(self, cls_score, label, b):
        # b, h, w = label.shape
        cls_result = torch.argmax(cls_score, dim=1).detach() + 1  # [b,h,w]
        label = label + 1
        diff = torch.eq(cls_result - label, 0).int().detach()  # [b,h,w] same:True,1; different:False,0
        invert_diff = 1 - diff  # bool can use ~
        b_ft_cross_region = [i for i in range(b)]
        b_tt_region = b_ft_cross_region.copy()
        b_cross_class = b_ft_cross_region.copy()
        b_tt_class = b_ft_cross_region.copy()
        for i in range(b):
            ft_cross_region = []
            tt_region = []
            tt_class = []
            cross_class = []
            # lists_labels_class: lists of labels' class
            lists_labels_class = set(torch.reshape(label[i, :, :], (1, -1)).cpu().numpy()[0].tolist())  # optimize
            for step, j in enumerate(lists_labels_class):
                if j == 256:
                    continue
                # tt: for class j, cls_result right classify,[h,w]
                tt = (cls_result[i, :, :] == j) * diff[i, :, :]  # bool
                # ft: for class j, cls_result false classify, may contain other class,[h,w]
                ft = (cls_result[i, :, :] == j) * invert_diff[i, :, :]  # bool
                cls_tt = cls_score[i, :, :, :] * tt
                cls_ft = label[i, :, :] * ft
                if len(tt_region) == 0:
                    tt_region = [cls_tt]
                    tt_class = [j]
                else:
                    tt_region.extend([cls_tt])
                    tt_class.append(j)
                var = set(torch.reshape(cls_ft, (1, -1)).cpu().numpy()[0].tolist())  # optimize
                # 网络输出为j，GT为k，ft_cross_region[cross_class]取得cross_class的位置
                # 先找出k类的位置，再找出错分为j的位置
                for k in var:
                    if k == 256 or k == 0:
                        continue
                    if len(ft_cross_region) == 0:
                        ft_cross_region = [(label[i, :, :] == k) * ft]  # [h,w]
                        cross_class = [str(j) + '/' + str(k)]
                    else:
                        ft_cross_region.extend([(label[i, :, :] == k) * ft])
                        cross_class.append(str(j) + '/' + str(k))
            try:
                tt_region = torch.stack(tt_region, dim=-1)  # [c, h, w, TT_class]
            except:
                tt_region = []
            try:
                ft_cross_region = torch.stack(ft_cross_region, dim=-1)  # [h,w,class]
            except:
                ft_cross_region = []
            b_ft_cross_region[i] = ft_cross_region
            b_cross_class[i] = cross_class
            b_tt_region[i] = tt_region
            b_tt_class[i] = tt_class

        return b_ft_cross_region, b_tt_region, b_cross_class, b_tt_class

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                original_cls_score=None,
                **kwargs):
        """Forward function belong cross region.

        Args:
            cls_score:[b,c,h,w]
            label:[b,h,w]
            weight:
            avg_factor:
            reduction_override:
            original_cls_score: [b,c,h/2,w/2]

        """
        self.temp = 10
        count = 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        b, c, _, _ = cls_score.shape
        # b_ft_cross_region1, b_tt_region1 = self.batch_cross_class(cls_score, label)
        b_ft_cross_region, b_tt_region_score, b_cross_class, b_tt_class = self.cross_class(cls_score, label, b)

        logits = [torch.zeros(1, device=cls_score.device) for b_logits in range(b)]
        for i in range(b):
            cross_class = b_cross_class[i]
            for j, key in enumerate(cross_class):
                class_key1, class_key2 = key.split('/', 1)
                tt_index1 = b_tt_class[i].index(int(class_key1))
                tt_index2 = b_tt_class[i].index(int(class_key2))

                phi1 = b_tt_region_score[i][:, :, :, tt_index1]
                phi2 = b_tt_region_score[i][:, :, :, tt_index2]
                phi1 = torch.reshape(phi1, (21, -1))
                phi2 = torch.reshape(phi2, (21, -1))
                phi1_light = phi1[:, phi1.sum(dim=0) != 0].detach()  # [21, length1]
                phi2_light = phi2[:, phi2.sum(dim=0) != 0].detach()  # [21, length2]
                cross_score = cls_score[i, :, :, :] * b_ft_cross_region[i][:, :, j]
                cross_score = torch.reshape(cross_score, (21, -1))
                cross_score_light = cross_score[:, cross_score.sum(dim=0) != 0]  # [21, length]
                phi1_length = phi1_light.shape[1]
                phi2_length = phi2_light.shape[1]
                cross_score_length = cross_score_light.shape[1]
                if phi1_length <= cross_score_length:
                    if phi1_length == 0:
                        continue
                    neg_scores = F.kl_div(cross_score_light[:, 0:phi1_length], phi1_light, reduction='mean')
                else:
                    neg_step = phi1_length // cross_score_length
                    neg_scores = torch.zeros(1, device=cross_score_light.device)
                    for nk in range(neg_step):
                        neg_scores_step = F.kl_div(cross_score_light,
                                                   phi1_light[:, nk * cross_score_length:(nk + 1) * cross_score_length],
                                                   reduction='mean')
                        neg_scores += neg_scores_step
                    neg_scores = neg_scores / neg_step

                if phi2_length <= cross_score_length:
                    if phi2_length == 0:
                        continue
                    pos_scores = F.kl_div(cross_score_light[:, 0:phi2_length], phi2_light, reduction='mean')
                else:
                    pos_step = phi2_length // cross_score_length
                    pos_scores = torch.zeros(1, device=cross_score_light.device)
                    for pk in range(pos_step):
                        pos_scores_step = F.kl_div(cross_score_light,
                                                   phi2_light[:, pk * cross_score_length:(pk + 1) * cross_score_length],
                                                   reduction='mean')
                        pos_scores += pos_scores_step
                    pos_scores = pos_scores / pos_step
                logits[i] += pos_scores - neg_scores
            if cross_class:
                logits[i] = logits[i] / len(cross_class)
                count += 1
        # log_path = '/data/lfxuan/projects/mmsegmentation/work_dirs/deeplabv3plus_r50-d8_512x512_20k_voc12aug_klpcl' \
        #            '/logits.log '
        if count == 0:
            logits = sum(logits)
        else:
            logits = sum(logits) / count
        loss2 = self.loss_weight * logits + loss_cls
        # print_log(f"seg_loss-{loss_cls.data},pwc_los-{logits.data}", logger=get_root_logger(log_file=log_path))
        return loss2

    def calc_logits(self, light_step, cross_score_light_step):

        # [e,21]*[21,length]--->[e,length]--->[length]

        # scores = torch.exp((light_step.T @ cross_score_light_step) / self.temp).sum(0)
        scores = torch.sum(torch.exp((light_step.T @ cross_score_light_step) / self.temp), dim=0)

        return scores


@LOSSES.register_module()
class KLPatchContrastiveLoss(nn.Module):
    """KLPatchContrastiveLossBatch contained CrossEntropyLoss.

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
                 loss_weight=0.1,
                 patch_size=16,
                 cal_function='KL'):
        super(KLPatchContrastiveLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.patch_size = patch_size
        self.temp = 50
        if cal_function == 'KL':
            self.cons_func = self.calculate_kl
        else:
            self.cons_func = self.calculate_cosin

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def cross_class(self, cls_score, label, b):
        # b, h, w = label.shape
        cls_result = torch.argmax(cls_score, dim=1).detach() + 1  # [b,h,w]
        label = label + 1
        diff = torch.eq(cls_result - label, 0).int()  # [b,h,w] same:True,1; different:False,0
        # invert_diff = (cls_result - label).nonzero()  # bool can use ~
        invert_diff = 1 - diff
        b_ft_cross_region = [i for i in range(b)]
        b_tt_region = b_ft_cross_region.copy()
        b_cross_class = b_ft_cross_region.copy()
        b_tt_class = b_ft_cross_region.copy()
        for i in range(b):
            ft_cross_region = []
            tt_region = []
            tt_class = []
            cross_class = []
            # lists_labels_class: lists of labels' class
            lists_labels_class = set(torch.reshape(label[i, :, :], (1, -1)).cpu().numpy()[0].tolist())  # optimize
            for step, j in enumerate(lists_labels_class):
                if j == 256:
                    continue
                # tt: for class j, cls_result right classify,[h,w]
                tt = (cls_result[i, :, :] == j) * diff[i, :, :]  # bool
                # ft: for class j, cls_result false classify, may contain other class,[h,w]
                ft = (cls_result[i, :, :] == j) * invert_diff[i, :, :]  # bool
                cls_tt = cls_score[i, :, :, :] * tt
                cls_ft = label[i, :, :] * ft
                if len(tt_region) == 0:
                    tt_region = [cls_tt]
                    tt_class = [j]
                else:
                    tt_region.extend([cls_tt])
                    tt_class.append(j)
                var = set(torch.reshape(cls_ft, (1, -1)).cpu().numpy()[0].tolist())  # optimize
                # 网络输出为j，GT为k，ft_cross_region[cross_class]取得cross_class的位置
                # 先找出k类的位置，再找出错分为j的位置
                for k in var:
                    if k == 256 or k == 0:
                        continue
                    if len(ft_cross_region) == 0:
                        ft_cross_region = [(label[i, :, :] == k) * ft]  # [h,w]
                        cross_class = [str(j) + '/' + str(k)]
                    else:
                        ft_cross_region.extend([(label[i, :, :] == k) * ft])
                        cross_class.append(str(j) + '/' + str(k))
            try:
                tt_region = torch.stack(tt_region, dim=-1)  # [c, h, w, TT_class]
            except:
                tt_region = []
            try:
                ft_cross_region = torch.stack(ft_cross_region, dim=-1)  # [h,w,class]
            except:
                ft_cross_region = []
            b_ft_cross_region[i] = ft_cross_region
            b_cross_class[i] = cross_class
            b_tt_region[i] = tt_region
            b_tt_class[i] = tt_class

        return b_ft_cross_region, b_tt_region, b_cross_class, b_tt_class

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                original_cls_score=None,
                **kwargs):
        """Forward function belong cross region.

        Args:
            cls_score:[b,c,h,w]
            label:[b,h,w]
            weight:
            avg_factor:
            reduction_override:
            original_cls_score: [b,c,h/2,w/2]

        """
        self.temp = 10
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        b, c, h, w = cls_score.shape
        _, _, h1, w1 = original_cls_score.shape
        cal_size = h1 * w1
        gt_seg = F.interpolate(label.unsqueeze(1).type(torch.cuda.FloatTensor), (h1, w1), mode='nearest').squeeze(1)
        gt_seg = gt_seg.type(torch.cuda.LongTensor)
        b_ft_cross_region, b_tt_region_score, b_cross_class, b_tt_class = self.cross_class(original_cls_score, gt_seg,
                                                                                           b)

        pos_feature = []
        neg_feature = []
        cross_feature = []

        for i in range(b):
            cross_class = b_cross_class[i]
            for j, key in enumerate(cross_class):
                class_key1, class_key2 = key.split('/', 1)
                tt_index1 = b_tt_class[i].index(int(class_key1))
                tt_index2 = b_tt_class[i].index(int(class_key2))
                phi1 = b_tt_region_score[i][:, :, :, tt_index1]
                phi2 = b_tt_region_score[i][:, :, :, tt_index2]
                phi1 = torch.reshape(phi1, (c, -1))
                phi2 = torch.reshape(phi2, (c, -1))
                phi1_light = phi1[:, phi1.sum(dim=0) != 0].detach()  # [21, length1]
                phi2_light = phi2[:, phi2.sum(dim=0) != 0].detach()  # [21, length2]
                cross_score = original_cls_score[i, :, :, :] * b_ft_cross_region[i][:, :, j]
                cross_score = torch.reshape(cross_score, (c, -1))
                cross_score_light = cross_score[:, cross_score.sum(dim=0) != 0]  # [21, length]
                phi1_length = phi1_light.shape[1]
                phi2_length = phi2_light.shape[1]
                cross_score_length = cross_score_light.shape[1]
                if phi1_length * phi2_length * cross_score_length == 0:
                    continue
                    ipdb.set_trace()

                reform_phi1_light = self.reform(phi1_light, phi1_length, cal_size, [h1, w1])
                reform_phi2_light = self.reform(phi2_light, phi2_length, cal_size, [h1, w1])
                reform_cross_light = self.reform(cross_score_light, cross_score_length, cal_size, [h1, w1])
                if not cross_feature:
                    pos_feature = [reform_phi2_light]
                    neg_feature = [reform_phi1_light]
                    cross_feature = [reform_cross_light]
                else:
                    pos_feature.append(reform_phi2_light)
                    neg_feature.append(reform_phi1_light)
                    cross_feature.append(reform_cross_light)
        count = len(pos_feature)
        if count != 0:
            pos_feature = torch.stack(pos_feature, dim=0)
            neg_feature = torch.stack(neg_feature, dim=0)
            cross_feature = torch.stack(cross_feature, dim=0)

            logits = self.cons_func(cross_feature, pos_feature, neg_feature, count)

            loss_cls = self.loss_weight * logits + loss_cls
        # print_log(f"seg_loss-{loss_cls.data},pwc_los-{logits.data}", logger=get_root_logger())
        return loss_cls

    def reform(self, phi_light, length, cal_size, target_shape):
        step = cal_size // length
        rest = cal_size % length
        reform_features = []
        for i in range(step):
            if not reform_features:
                reform_features = [phi_light]
            else:
                reform_features.append(phi_light)
        reform_features.append(phi_light[:, :rest])
        reform_features = torch.cat(reform_features, dim=1)
        reform_features = torch.reshape(reform_features, (-1, target_shape[0], target_shape[1]))
        return reform_features

    def calculate_kl(self, cross_feature, pos_feature, neg_feature, count):
        pos_scores = F.kl_div(cross_feature, pos_feature, reduction='mean')
        neg_feature = F.kl_div(cross_feature, neg_feature, reduction='mean')
        pos_scores = pos_scores if pos_scores != 0 else 1
        logits = -torch.log(pos_scores / (pos_scores + neg_feature + 1e-8)) / count
        return logits

    def calculate_cosin(self, cross_feature, pos_feature, neg_feature, count):
        pos_scores = F.cosine_similarity(cross_feature, pos_feature, dim=0).sum()
        neg_feature = F.cosine_similarity(cross_feature, neg_feature, dim=0).sum()
        ipdb.set_trace()
        pos_scores = pos_scores if pos_scores != 0 else 1
        logits = -torch.log(pos_scores / (pos_scores + neg_feature + 1e-8)) / count
        return logits
