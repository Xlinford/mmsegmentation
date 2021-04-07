_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-cospcl.py',
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True))
test_cfg = dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
# optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0005)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU')