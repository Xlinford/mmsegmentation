_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8-klpcl.py',
    '../../_base_/datasets/cityscapes.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(loss_decode=dict(type='KLPatchContrastiveLoss',
                                      use_sigmoid=False,
                                      loss_weight=0.1,
                                      cal_function='COS')))
# optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0005)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU')