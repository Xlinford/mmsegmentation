_base_ = '../deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             decode_head=dict(loss_decode=dict(type='KLPatchContrastiveLoss',
                                               use_sigmoid=False,
                                               loss_ratio=True,
                                               loss_weight=1,
                                               cal_function='COS07',
                                               cal_gate=[20, 199])))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
log_config = dict(interval=50,
                  hooks=[dict(type='TextLoggerHook', by_epoch=False),
                         dict(type='TensorboardLoggerHook')])
