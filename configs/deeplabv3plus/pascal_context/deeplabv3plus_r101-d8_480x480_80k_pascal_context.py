_base_ = '../deeplabv3plus_r50-d8_480x480_80k_pascal_context.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             decode_head=dict(loss_decode=dict(type='KLPatchContrastiveLoss',
                                               use_sigmoid=False,
                                               loss_weight=0.1,
                                               cal_function='COS07',
                                               cal_gate=99)))
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU')
