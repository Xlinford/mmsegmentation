import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def corner_crop(crop_region,
                feat,
                pseudo_labels,
                repeat_times=2):
    feats_crop = []
    labels_crop = []
    for i in range(repeat_times):
        feat_one = feat[:, 128*i:(128*(i+1)), :, :]
        pseudo_label_one = pseudo_labels[:, 21*i:(21*(i+1)), :, :]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_region[i]
        feats_crop[i] = feat_one[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        labels_crop[i] = pseudo_label_one[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    feats_return = feats_crop[0]
    labels_return = labels_crop[0]

    for i in range(len(feats_crop)-1):
        feats_return = torch.cat((feats_return, feats_crop[i+1]), dim=1)
        labels_return = torch.cat((labels_return, labels_crop[i+1]), dim=1)

    return feats_return, labels_return


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)
