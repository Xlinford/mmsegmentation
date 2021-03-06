from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import ipdb
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize, corner_crop
from ..builder import build_loss
from ..losses import accuracy


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        # if isinstance(loss_decode, list):
        #     self.loss_decode = build_loss(loss_decode[0])
        #     self.loss_decode1 = build_loss(loss_decode[1])
        # else:
        #     self.loss_decode = build_loss(loss_decode)
        #     self.loss_decode1 = None
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, inputs1=None, inputs2=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
            inputs1(list[Tensor]): List of multi-level img features.
            inputs2(list[Tensor]): List of multi-level img features.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if inputs2 is not None:
            seg_logits, seg_logits_concat, seg_label = self.forward([inputs, inputs1, inputs2])
            # seg_logits1, feat1 = self.forward(inputs1, mlp=True)
            # seg_logits2, feat2 = self.forward(inputs2,  mlp=True)
            # seg_label = torch.cat((seg_logits1, seg_logits2), dim=1)
            # seg_logits_concat = torch.cat((feat1, feat2), dim=1)
            losses = self.contrastive_losses(seg_logits, gt_semantic_seg, seg_logits_concat, seg_label, img_metas)

        else:
            seg_logits = self.forward(inputs)
            losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit1 = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit1, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss_seg_tuple = self.loss_decode(
            seg_logit1,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index,
            original_cls_score=seg_logit)
        if type(loss_seg_tuple) == tuple:
            loss['loss_seg'] = loss_seg_tuple[0]
            loss['conloss_seg'] = loss_seg_tuple[1]
        else:
            loss['loss_seg'] = loss_seg_tuple
        loss['acc_seg'] = accuracy(seg_logit1, seg_label)
        return loss

    @force_fp32(apply_to=('seg_logits', 'seg_logits1', 'seg_label'))
    def contrastive_losses(self, seg_logits, gt_semantic_seg, seg_logits1, seg_label, img_metas):
        """Compute pixel-wise contrastive loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logits,
            size=gt_semantic_seg.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        for i, logit in enumerate(seg_logits1):
            seg_logits1[i] = resize(
                input=logit,
                size=img_metas[0]['img_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners)

        for i, label in enumerate(seg_label):
            seg_label[i] = resize(
                input=label,
                size=img_metas[0]['img_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, gt_semantic_seg)
        else:
            seg_weight = None
        gt_semantic_seg = gt_semantic_seg.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logits1,
            seg_label,
            seg_logit,
            gt_semantic_seg,
            img_metas,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        # loss['loss_seg'] = self.loss_decode1(
        #                     seg_logits1,
        #                     seg_label,
        #                     img_metas,
        #                     weight=None,
        #                     ignore_index=self.ignore_index)
        # loss['loss_seg'] = self.loss_decode1(
        #     seg_logits1,
        #     seg_label,
        #     img_metas,
        #     weight=None,
        #     ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, gt_semantic_seg)

        return loss
