_base_ = './deeplabv3plus_r50-d8_512x1024_80k_cityscapes_cos07pcl.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             decode_head=dict(loss_decode=dict(type='KLPatchContrastiveLoss',
                                               use_sigmoid=False,
                                               loss_weight=0.1,
                                               cal_function='COS07',
                                               cal_gate=99)))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)